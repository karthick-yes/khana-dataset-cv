Khana: A Comprehensive Indian Cuisine Dataset
Description
Khana is a food classification dataset featuring a wide range of dishes from Indian cuisine. The goal of this dataset is to solve challenges found in existing datasets: lack of representation of Indian food dishes, generalization over limited diversity due to Western-influenced classes, conditions, viewpoints and environments. The dataset contains 131,000+ images comprising 80 food dishes. You can download the dataset, labels from the dataset, and the taxonomy created during its inception here. Khana does not own the copyright of the images. It only compiles an accurate list of web images for each food dish. It is available for researchers and educators who wish to use the images for non-commercial research and/or educational purposes only.

Detailed Statistics
The figure shows the detailed statistics of Khana with distributions of categories and images per food category.

Dataset and Additional Links
Research Article: https://arxiv.org/pdf/2509.06006
Dataset GitHub: https://github.com/prabhuomkar/Khana
Dataset Link: https://drive.google.com/drive/folders/1PWyJdkizw5ABBd8BIAnr_FZq91YZ2Uo0?usp=sharing
Dataset Website: https://khana.omkar.xyz/

Problems to be solved – 3 +1 Bonus, 33%+20% (Bonus)

Image Classification – Design and train a classifier to achieve better classification performance (over 80 classes) than baseline. Marks will be awarded based on the leaderboard. Baseline validation accuracy (80% train, 20% validation) is ~91%. Below the baseline validation accuracy will get 0%, and will not be evaluated further. If validation accuracy is above baseline, then we will provide 20-30 test images on which the final test accuracy will be computed. The leader will get 100% of marks, next 95% and so on.

There are clean thali images such as this

You have to run detection algorithm on this such that you can get an output like this

However note that the labels are incorrect in this picture – they do not belong to the 80 classes in the Khana dataset. Develop a method such that the detection results give the correct labels and bounding boxes.

We will evaluate this using a held out test set – but only with respect to the labels, not the accuracy of the bounding box. Precision Recall will be used as the metric.

Instead of above clean images you will be given natural images like below, with detection (also shown below) suffering because the view is not a clean bird's eye view. Develop a method to get a BEV of these natural images and then run the detection algorithm in 2. Evaluation here will be qualitative – visual inspection.

BONUS – use the steps in https://cal-cs180.github.io/fa25/hw/proj4/index.html to create a NRF of your thaali.
