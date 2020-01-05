from methods.base import BaseClassifier
from methods.standard import StandardClassifier, StandardClassifierWithNoise
from methods.penalize import PenalizeLastLayerFixedForm, PenalizeLastLayerGeneralForm
from methods.predict import PredictGradOutput, PredictGradOutputFixedFormWithConfusion,\
    PredictGradOutputGeneralFormUseLabel
from methods.vae import VAE
