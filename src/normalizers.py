from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class normalizers:
    
    def __init__(self, classifier):
        self.classifier = classifier
        
    def no_normalization(self):
        return self.classifier
    
    def min_max_normalization(self):
        return Pipeline([
            ('mms', MinMaxScaler()),
            ('logistic', self.classifier)
        ])
    
    def standardization(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', self.classifier)
        ])
    
    def get_all(self):
        return {
            "No Normalization": self.no_normalization(),
            "Min-Max Normalization": self.min_max_normalization(),
            "Feature Standardization": self.standardization()
        }