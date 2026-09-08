# --------------------------------------------------------
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model
# By Jiarui Jin and Haoyu Wang
# Based on BEiT-v2, timm, DeiT, DINO and LaBraM code bases
# ---------------------------------------------------------

"""用于 HeartLang 微调阶段的 sklearn 风格特征模型。"""

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 可选依赖：如果环境里没装 xgboost，则置为 None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# 可选依赖：如果环境里没装 lightgbm，则置为 None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


# 支持的 sklearn 风格特征分类模型类型
SKLEARN_FEATURE_MODEL_TYPES = [
    "decision_tree",
    "random_forest",
    "svm",
    "logistic_regression",
    "xgboost",
    "lightgbm",
]


def get_positive_class_index(class_labels):
    """
    在类别标签中找到正类（标签值为 1）对应的下标。
    常用于二分类任务里，从 predict_proba 的输出中取“正类概率”。

    参数:
        class_labels: 分类器内部保存的类别标签列表/数组

    返回:
        正类 1 对应的列下标；
        如果没有显式找到 1，则退化为返回 1 或最后一个合法下标
    """
    class_labels = np.asarray(class_labels)
    positive_matches = np.where(class_labels == 1)[0]
    if len(positive_matches) > 0:
        return int(positive_matches[0])
    return min(1, len(class_labels) - 1)


def normalize_classifier_probabilities(probabilities, classes, target_template, is_binary):
    """
    将不同 sklearn 分类器输出的概率格式统一整理。

    作用:
    1. 多分类时：直接转成 numpy 数组返回
    2. 二分类时：统一提取“正类概率”
    3. 兼容不同模型/不同任务下概率输出 shape 不一致的问题
    4. 尽量对齐 target_template 的形状

    参数:
        probabilities: 分类器 predict_proba 的原始输出
        classes: 分类器的类别标签（如 classifier.classes_）
        target_template: 用来参考目标张量 shape 的模板
        is_binary: 是否为二分类任务

    返回:
        规范化后的概率数组
    """
    # 多分类不做特殊处理，直接返回
    if not is_binary:
        return np.asarray(probabilities)

    # 某些情况下 probabilities 可能是 list（例如多标签或多输出分类器）
    if isinstance(probabilities, list):
        positive_probs = []
        for idx, class_probs in enumerate(probabilities):
            class_probs = np.asarray(class_probs)
            # 如果 classes 是 list/tuple，则每个输出头有一份 classes
            class_labels = classes[idx] if isinstance(classes, (list, tuple)) else classes

            # 一维情况：默认它已经是正类概率
            if class_probs.ndim == 1:
                positive_probs.append(class_probs.astype(np.float32))

            # 只有一列概率：说明该输出可能只预测到单一类别
            elif class_probs.shape[1] == 1:
                if class_labels is None:
                    positive_probs.append(class_probs[:, 0].astype(np.float32))
                else:
                    # 如果唯一类别就是 1，则正类概率恒为 1；否则恒为 0
                    positive_value = 1.0 if int(np.asarray(class_labels)[0]) == 1 else 0.0
                    positive_probs.append(np.full(class_probs.shape[0], positive_value, dtype=np.float32))

            # 正常二分类两列以上：[P(class0), P(class1)]，取正类那一列
            else:
                positive_probs.append(class_probs[:, get_positive_class_index(class_labels)])

        # 多个输出头拼成 [N, num_outputs]
        probabilities = np.stack(positive_probs, axis=1)

    else:
        probabilities = np.asarray(probabilities)

        # 三维情况：通常取最后一维中的正类概率
        if probabilities.ndim == 3:
            probabilities = probabilities[..., 1]

        # shape=[N,1]：只有一个类别概率
        elif probabilities.ndim == 2 and probabilities.shape[1] == 1:
            if classes is None:
                probabilities = probabilities[:, 0]
            else:
                class_labels = np.asarray(classes)
                positive_value = 1.0 if int(class_labels[0]) == 1 else 0.0
                probabilities = np.full(probabilities.shape[0], positive_value, dtype=np.float32)

        # shape=[N,2] 或更多：取正类所在列
        elif probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            positive_index = get_positive_class_index(classes) if classes is not None else 1
            probabilities = probabilities[:, positive_index]

    # 如果 target 是二维单列，而当前概率是一维，则补成 [N,1]
    target_template = np.asarray(target_template)
    if target_template.ndim > 1 and target_template.shape[1] == 1 and probabilities.ndim == 1:
        probabilities = probabilities[:, None]

    return probabilities


class SklearnFeatureModel:
    """
    对 sklearn 风格分类器做一层统一封装，
    让它在 HeartLang 的训练/评估流程里使用方式更一致。
    """

    def __init__(self, args, classifier, model_name, prepare_targets_fn):
        """
        参数:
            args: 训练配置参数
            classifier: 具体 sklearn 分类器实例
            model_name: 模型名，用于保存产物
            prepare_targets_fn: 目标预处理函数，用于把 targets 转成 sklearn 可接受的格式
        """
        self.args = args
        self.classifier = classifier
        self.model_name = model_name
        self.artifact_name = f"{model_name}.pkl"  # 保存时的文件名
        self.prepare_targets_fn = prepare_targets_fn

    def fit(self, train_data, val_data=None):
        """
        用训练集特征和标签训练分类器。
        这里 val_data 没有使用，只是为了和外部统一接口兼容。
        """
        del val_data
        self.classifier.fit(
            train_data["features"],
            self.prepare_targets_fn(train_data["targets"], self.args.is_binary),
        )

    def predict(self, features):
        """返回类别预测结果。"""
        return self.classifier.predict(features)

    def predict_proba(self, features, target_template=None):
        """
        返回概率预测，并进一步做格式统一。
        target_template 用来帮助对齐输出 shape。
        """
        probabilities = self.classifier.predict_proba(features)
        return normalize_classifier_probabilities(
            probabilities,
            getattr(self.classifier, "classes_", None),
            target_template,
            self.args.is_binary,
        )

    def compute_loss(self, features, targets):
        """
        sklearn 分类器通常不在这里显式计算 loss，
        因此返回 None，仅保留接口兼容性。
        """
        del features, targets
        return None

    def save(self, output_dir):
        """将训练好的分类器保存为 pkl 文件。"""
        with open(output_dir / self.artifact_name, "wb") as f:
            pickle.dump(self.classifier, f)


def create_sklearn_feature_estimator(args, feature_model_type):
    """
    根据 feature_model_type 创建对应的 sklearn / boosting 分类器实例。
    """

    if feature_model_type == "decision_tree":
        # 决策树分类器
        return DecisionTreeClassifier(
            criterion=args.dt_criterion,          # 划分标准，如 gini / entropy
            max_depth=args.dt_max_depth,          # 树最大深度
            random_state=args.dt_random_state,    # 随机种子
        )

    if feature_model_type == "random_forest":
        # 随机森林分类器
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,    # 树的数量
            max_depth=args.rf_max_depth,          # 每棵树最大深度
            random_state=args.rf_random_state,    # 随机种子
            n_jobs=args.rf_n_jobs,                # 并行线程数
        )

    if feature_model_type == "svm":
        # SVM 分类器
        return SVC(
            kernel=args.svm_kernel,               # 核函数，如 linear / rbf
            C=args.svm_c,                         # 正则化强度
            gamma=args.svm_gamma,                 # 核函数参数
            probability=True,                     # 允许输出 predict_proba
            random_state=args.svm_random_state,   # 随机种子
        )

    if feature_model_type == "logistic_regression":
        # 逻辑回归分类器
        return LogisticRegression(
            max_iter=args.lr_max_iter,            # 最大迭代次数
            C=args.lr_c,                          # 正则化强度
            solver=args.lr_solver,                # 优化器
            random_state=args.lr_random_state,    # 随机种子
        )

    if feature_model_type == "xgboost":
        # 如果没装 xgboost，则报错
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed, but feature_model_type=xgboost was requested.")

        # XGBoost 通用参数
        params = {
            "n_estimators": args.xgb_n_estimators,              # 弱学习器数量
            "max_depth": args.xgb_max_depth,                    # 树深度
            "learning_rate": args.xgb_learning_rate,            # 学习率
            "subsample": args.xgb_subsample,                    # 行采样比例
            "colsample_bytree": args.xgb_colsample_bytree,      # 列采样比例
            "random_state": args.xgb_random_state,              # 随机种子
            "n_jobs": args.xgb_n_jobs,                          # 并行线程数
            "verbosity": 0,                                     # 静默输出
        }

        # 根据二分类 / 多分类任务设置目标函数
        if args.is_binary:
            params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": args.nb_classes})

        return XGBClassifier(**params)

    if feature_model_type == "lightgbm":
        # 如果没装 lightgbm，则报错
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed, but feature_model_type=lightgbm was requested.")

        # LightGBM 通用参数
        params = {
            "n_estimators": args.lgbm_n_estimators,         # 树数量
            "learning_rate": args.lgbm_learning_rate,       # 学习率
            "num_leaves": args.lgbm_num_leaves,             # 叶子节点数
            "max_depth": args.lgbm_max_depth,               # 最大深度
            "random_state": args.lgbm_random_state,         # 随机种子
            "n_jobs": args.lgbm_n_jobs,                     # 并行线程数
            "verbosity": -1,                                # 静默输出
        }

        # 根据任务类型设置目标函数
        if args.is_binary:
            params["objective"] = "binary"
        else:
            params.update({"objective": "multiclass", "num_class": args.nb_classes})

        return LGBMClassifier(**params)

    # 如果传入了不支持的模型类型，直接报错
    raise ValueError(f"Unsupported sklearn feature model type: {feature_model_type}")


def build_sklearn_feature_model(args, feature_model_type, prepare_targets_fn):
    """
    先创建具体分类器，再封装成统一的 SklearnFeatureModel。
    """
    classifier = create_sklearn_feature_estimator(args, feature_model_type)
    return SklearnFeatureModel(args, classifier, feature_model_type, prepare_targets_fn)