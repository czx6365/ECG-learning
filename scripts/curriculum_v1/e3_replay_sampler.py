"""Deterministic ECG-level patient-heavy replay, with no within-epoch repeats."""
import numpy as np
import torch
from torch.utils.data import Sampler


class PatientReplaySampler(Sampler):
    def __init__(self, patient_indices, healthy_indices, batch_size=256, seed=0):
        self.patient = np.asarray(patient_indices, dtype=np.int64)
        self.healthy = np.asarray(healthy_indices, dtype=np.int64)
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        if not len(self.patient) or not len(self.healthy) or batch_size < 3:
            raise ValueError('Both cohorts and batch_size >= 3 are required')
        if len(np.unique(self.patient)) != len(self.patient) or len(np.unique(self.healthy)) != len(self.healthy):
            raise ValueError('Repeated pool indices')
        if np.intersect1d(self.patient, self.healthy).size:
            raise ValueError('Cohort pools overlap')
        self.steps = 0
        while self._patient_count(self.steps + 1) <= len(self.patient):
            self.steps += 1
        if self.steps == 0:
            raise ValueError('Too few patients for a complete batch')
        self.patient_count = self._patient_count(self.steps)
        self.healthy_count = self.steps * batch_size - self.patient_count
        if self.healthy_count > len(self.healthy):
            raise ValueError('Too few healthy records for replay without replacement')

    def _patient_count(self, steps):
        return (2 * steps * self.batch_size + 1) // 3

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps * self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        patient = self.patient[torch.randperm(len(self.patient), generator=generator).numpy()][:self.patient_count]
        healthy = self.healthy[torch.randperm(len(self.healthy), generator=generator).numpy()][:self.healthy_count]
        pi = hi = 0
        for step in range(self.steps):
            npatient = self._patient_count(step + 1) - self._patient_count(step)
            nhealthy = self.batch_size - npatient
            batch = np.concatenate([patient[pi:pi+npatient], healthy[hi:hi+nhealthy]])
            pi += npatient
            hi += nhealthy
            yield from batch[torch.randperm(self.batch_size, generator=generator).numpy()].tolist()
