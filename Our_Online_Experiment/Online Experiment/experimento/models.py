# experimento/models.py

import json
import numpy as np
import uuid
from django.db import models
from experimento.utils import Experiment, Prior

class ExperimentSession(models.Model):
    participant_id = models.CharField(max_length=100)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    last_response = models.CharField(max_length=1, null=True, blank=True)
    correct_responses = models.IntegerField(null=True, blank=True)  # Respuestas correctas
    email = models.EmailField(null=True, blank=True)  # Correo electrónico
    prior_type = models.IntegerField(null=True, blank=True)  # Tipo previo
    corr_trial = models.IntegerField(default=0)
    age = models.CharField(max_length=20, null=True, blank=True)  # Rango de edad
    gender = models.CharField(max_length=20, null=True, blank=True)  # Género
    education = models.CharField(max_length=50, null=True, blank=True)  # Nivel educativo
    num_list = models.JSONField(default=list)
    experiment_data = models.JSONField(null=True, blank=True)  # Aquí se almacenará el objeto serializado
    ff=models.JSONField(default=list)
    random_trials=models.JSONField(default=list)
class ExperimentManager:
    def __init__(self):
        self.experiment = None

    def get_experiment(self, num):
        if num == 0:
            self.experiment = Experiment()
            return self.experiment
        else:
            if self.experiment is not None:
                return self.experiment
            else:
                raise ValueError("El experimento no ha sido inicializado todavía.")
            
    def serialize_experiment(self):
        if self.experiment:
            return json.dumps(self.experiment.__dict__, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        else:
            raise ValueError("No hay experimento para serializar.")

    def deserialize_experiment(self, data):
        self.experiment = Experiment()
        self.experiment.__dict__ = json.loads(data)
        for key, value in self.experiment.__dict__.items():
            if isinstance(value, list):
                self.experiment.__dict__[key] = np.array(value)


    
class Block(models.Model):
    session = models.ForeignKey(ExperimentSession, on_delete=models.CASCADE)
    block_number = models.IntegerField()
    performance = models.FloatField(null=True, blank=True)

class Trial(models.Model):
    id = models.BigAutoField(primary_key=True)
    block = models.ForeignKey('experimento.Block', on_delete=models.CASCADE)
    trial_id = models.IntegerField()  # Use trial_id if that’s your requirement
    stimulus = models.CharField(max_length=10)
    response = models.CharField(max_length=10, null=True, blank=True)
    d1= models.IntegerField(default=1)
    reaction_time = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    participant_id = models.CharField(max_length=100)  # Agrega este campo
    ado_type=models.IntegerField(default=1)
    corr_esp_trial = models.IntegerField(null=True, blank=True)
