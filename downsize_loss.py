import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi
import cv2 as cv
import matplotlib.pyplot as plt

def size_reducer(img, biggest_length):
    h, w = img.shape[:2]
    if max(h, w) > biggest_length:
        if h > w:
            new_h = biggest_length
            new_w = int(w * biggest_length / h)
        else:
            new_w = biggest_length
            new_h = int(h * biggest_length / w)
        img = cv.resize(img, (new_w, new_h))
    return img

def resized_imgs(img, smallest_length=50, increment=10):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    sizes = list(range(smallest_length, max_dim + 1, increment))
    resized_list = [size_reducer(img, size) for size in sizes]
    reresized_list = [cv.resize(rimg, (w, h)) for rimg in resized_list]
    return reresized_list, sizes

def main(args):
    if args.grayscale:
        img = cv.imread(args.img, cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(args.img, cv.IMREAD_GRAYSCALE)

    # resized = size_reducer(img, args.downsize)
    # reresized = cv.resize(resized, (img.shape[1], img.shape[0]))

    # ssim_loss = ssim(img, reresized, multichannel=not args.grayscale)

    reresized_list, sizes = resized_imgs(img, smallest_length=50, increment=10)
    ssim_losses = [ssim(img, rimg, multichannel=not args.grayscale) for rimg in reresized_list]
    nmi_losses = [nmi(img, rimg) for rimg in reresized_list]
    
    plt.figure(figsize=(10,5))
    plt.plot(sizes, ssim_losses, marker='o', label='SSIM')
    plt.plot(sizes, nmi_losses, marker='s', label='NMI')
    plt.title("Image Similarity Metrics vs Downsize Dimension")
    plt.xlabel("Downsize Dimension (pixels)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid()

    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.title(f"Original {img.shape[1]}x{img.shape[0]}")
    # plt.imshow(img, cmap='gray' if args.grayscale else None)
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.title(f"Downsized {resized.shape[1]}x{resized.shape[0]} (SSIM: {ssim_loss:.2f})")
    # plt.imshow(resized, cmap='gray' if args.grayscale else None)
    # plt.axis('off')
    # plt.show()

    plt.savefig("downsize_loss.png")
    plt.show()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Input file path")
    parser.add_argument("--downsize", type=int, default=50, help="Largest pixel dimension after downsize")
    parser.add_argument("--grayscale", action='store_true', default=True, help="Convert image to grayscale")

    args = parser.parse_args()
    main(args)