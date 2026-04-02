import nibabel as nib

image_path="/mnt/data/ilinca/structured_cases_14_16/D6FB61B39B2A80/image.nii.gz"
mask_path="/mnt/data/ilinca/structured_cases_14_16/D6FB61B39B2A80/mask_CTV-aksilperikl.nii.gz"
img = nib.load(image_path)
msk = nib.load(mask_path)

print("image shape:", img.shape)
print("mask shape:", msk.shape)
print("image affine:\n", img.affine)
print("mask affine:\n", msk.affine)