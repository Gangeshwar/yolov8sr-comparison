import os
import shutil

resultsPath = 'runs/train'

if os.path.exists(resultsPath):   
    shutil.copy(f"{resultsPath}/weights/best.pt", ".")
    shutil.copy(f"{resultsPath}/P_curve.png", ".")
    shutil.copy(f"{resultsPath}/results.csv", ".")
    shutil.copy(f"{resultsPath}/train_batch0.jpg", ".")
    shutil.copy(f"{resultsPath}/val_batch0_labels.jpg", ".")
    shutil.copy(f"{resultsPath}/val_batch0_pred.jpg", ".")
    shutil.copy(f"{resultsPath}/labels_correlogram.jpg", ".")
    shutil.copy(f"{resultsPath}/labels.jpg", ".")
    shutil.rmtree('runs')









