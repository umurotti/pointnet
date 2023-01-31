import numpy as np


def main():
    encoding_path1 = 'input/bed1.npy'
    encoding_path2 = 'input/bed2.npy'

    with open(encoding_path1, 'rb') as file1, open(encoding_path2, 'rb') as file2:
        encoding1 = np.load(file1)
        encoding2 = np.load(file2)
        l2_loss = np.linalg.norm(encoding1 - encoding2)
        print(f'L2 loss between {encoding_path1} and {encoding_path2}: {l2_loss}')

if __name__ == '__main__':
    main()