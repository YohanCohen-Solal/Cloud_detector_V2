<h1>
    Accuracy calculations:
</h1>


    for i in range(len(files1)):
        image = Image.open(os.path.join(IMAGE_PATH, files1[i]))
        pred = Image.open(os.path.join(PRED_PATH, files2[i]))

        if image.size != pred.size:
            raise ValueError(f"Image size mismatch: {files1[i]} vs {files2[i]}")

        np_image = np.array(image)
        np_pred = np.array(pred)

        num_pixel_total += np_image.size
        num_pred_correct += np.sum(np_pred == np_image)

    accuracy = 100.0 * num_pred_correct / num_pixel_total
    print("Accuracy: {:.2f}%".format(accuracy))


<p>
This fuction goes through all images in the pred and saved_images and takes all the pixel in the Ground truth(images inside saved_images) and then increments by one each time the predicted image has the same pixel as the ground truth
</p>


    def check_accuracy(loader, model, device="cuda"):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                y[y == 100] = 0

                preds = torch.sigmoid(model(x)) # Prédit la segmentation
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        print(
            f"Got {num_correct}/{num_pixels} with acc {(num_correct/num_pixels)*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(loader)}")
        model.train()

<p>
The accuracy of the model's segmentation predictions is calculated by dividing the number of pixels that were correctly classified by the total number of pixels in the input images. To compute the number of correctly classified pixels, the function compares each pixel in the predicted segmentation mask to the corresponding pixel in the ground truth segmentation mask, and counts the number of pixels for which the two masks match. This count is stored in the num_correct variable. The total number of pixels is obtained by multiplying the height and width of the input images together, and this count is stored in the num_pixels variable. Finally, the accuracy is computed by dividing num_correct by num_pixels, and multiplying the result by 100 to obtain a percentage.
</p>