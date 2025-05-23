import scipy.io
import os

def convert_mat_to_txt(mat_path, image_folder, output_txt):
    mat = scipy.io.loadmat(mat_path)
    data = None
    # Pilih key yang ada di file mat
    if 'testdata' in mat:
        data = mat['testdata'][0]
    elif 'traindata' in mat:
        data = mat['traindata'][0]
    else:
        raise ValueError("File .mat tidak mengandung 'testdata' atau 'traindata'")

    with open(output_txt, 'w', encoding='utf-8') as f_out:
        for item in data:
            img_name = item[0][0]  # contoh: 'test/2562_1.png' atau 'train/2562_1.png'
            label = item[1][0]
            label = label.strip().lower()

            # Ambil nama file saja tanpa folder 'train/' atau 'test/'
            filename = os.path.basename(img_name)
            path = os.path.join(image_folder, filename).replace("\\", "/")

            f_out.write(f"{path}\t{label}\n")

    print(f"✅ Label file created: {output_txt}")

if __name__ == "__main__":
    convert_mat_to_txt(
        mat_path="IIIT5K/testdata.mat",
        image_folder="dataset/val_images",
        output_txt="dataset/val_labels.txt"
    )

    convert_mat_to_txt(
        mat_path="IIIT5K/traindata.mat",
        image_folder="dataset/train_images",
        output_txt="dataset/train_labels.txt"
    )
