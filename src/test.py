import tifffile

pm = tifffile.imread(r"C:\Users\User\Documents\GitHub\ST\models\eyes_probs.tiff")
print(pm.shape, pm.dtype)