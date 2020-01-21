# Project: 3D Object Tracking

[//]: # (Image References)
[gif_intro]: ./media/intro.gif
[img_lidar_table]: ./media/LidarTTCManual.png
[img_matches_num_table]: ./media/number_of_matches_table.png
[img_lidar_outlier]: ./media/lidar_outlier.png
[img_udacity_lidarttc]: ./media/udacity_lidarttc.png
[img_udacity_camerattc1]: ./media/udacity_camerattc1.png
[img_udacity_camerattc2]: ./media/udacity_camerattc2.jpg
[plot_lidar]: ./media/plot_lidarttc.png
[plot_akaze]: ./media/plot_akaze.png
[plot_all_combinations]: ./media/plot_all_combinations.png
[plot_brisk]: ./media/plot_brisk.png
[plot_detection_overview]: ./media/plot_detection_overview.png
[plot_fast]: ./media/plot_fast.png
[plot_harris]: ./media/plot_harris.png
[plot_orb]: ./media/plot_orb.png
[plot_shitomasi]: ./media/plot_shitomasi.png
[plot_sift]: ./media/plot_sift.png
[table_all_combinations]: ./media/table_all_combinations.png
[img_bump1]: ./media/bumper1.png
[img_bump2]: ./media/bumper2.png
[img_topv1]: ./media/lidar_topview5.png
[img_topv2]: ./media/lidar_topview6.png
[img_topv3]: ./media/lidar_topview7.png



## Final Report

This report summarizes the work done for the 3D Object Tracking Project in the Sensor Fusion Nanodegree from Udacity.
The main goal is to compare the performance of different keypoint detectors and track those points across successive images.
Instead of providing a comprehensive explanation, this document is organized according to the rubric points required for the project.

![][gif_intro]

### FP.0 Final Report

> Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.

Each section of this document addresses the corresponding rubric points.


### FP.1 Match 3D Objects

> Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

The matching between two bounding boxes is done using an auxiliary matrix where each column and row corresponds to the boxID of the previous and the current data frames.
Each cell represents the number of keypoint matches between each boxID and it is computed by iterating over all possible combinations of two bounding boxes.
After the matrix is calculated, the pairs of matching bounding boxes are selected based on the maximum value for each row. The result is then stored in the `bbBestMatches` structure.


The code for this function is displayed below.

```cpp
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //std::cout << bbBestMatches << std::endl;
    //Loop through each match
    int currBoxCount = currFrame.boundingBoxes.size();
    int prevBoxCount = prevFrame.boundingBoxes.size();
    int max [currBoxCount][prevBoxCount] = { }; // Initialized with zeros

    // For each match, loop through the bounding boxes of current and previous frame to find which bounding boxes contains it.
    for (const cv::DMatch& match : matches)
    {
        for (const BoundingBox& bbCurr : currFrame.boundingBoxes)
        {
            for (const BoundingBox& bbPrev : prevFrame.boundingBoxes)
            {
                if (bbCurr.roi.contains(currFrame.keypoints[match.trainIdx].pt) &&
                    bbPrev.roi.contains(prevFrame.keypoints[match.queryIdx].pt))
                {
                    max[bbCurr.boxID][bbPrev.boxID] += 1;
                }
            }
        }
    }


    // For each box in the current frame, select the box from previous frame with the highest number of matches
    for (const BoundingBox& bbCurr : currFrame.boundingBoxes)
    {
        int maximum = 0;
        int index = 0;

        for (const BoundingBox& bbPrev : prevFrame.boundingBoxes)
        {
            if (maximum < max[bbCurr.boxID][bbPrev.boxID])
            {
                maximum = max[bbCurr.boxID][bbPrev.boxID];
                index = bbPrev.boxID;
            }
        }
        bbBestMatches.insert (std::pair<int,int>(index,bbCurr.boxID));
    }
}
```




### FP.2 Compute Lidar-based TTC

> Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

The Lidar sensor data can be used to compute the time to collision. The image below (from Udacity sensor fusion course) illustrates the concept.

![][img_udacity_lidarttc]

To compute the time-to-collision, first, the lidar points are organized in vectors order by the X position. It allows to easily find the points that are closer to the ego vehicle at the beginning of the vector. After sorting the lidar points it is possible to observe a few outliers that do not  represent a valid information from the car as it can be seen in bottom-right edge of the bounding box in the image below.

![][img_lidar_outlier]

Inspecting all the available data, it is possible to verify that the maximum number of outliers is always one. Therefore, selecting the second point in the array is enough to obtain stable results in this test. For real application though, a more robust way to eliminate the outliers should be implemented.

The Lidar-based time-to-collision is computed in the function below.


```cpp
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    //Organize x values in a sorted array
    vector<double> prevXPoints, currXPoints;

    for (const LidarPoint& point : lidarPointsPrev)
    {
        prevXPoints.push_back(point.x);
    }
    std::sort(prevXPoints.begin(), prevXPoints.end());

    for (const LidarPoint& point : lidarPointsCurr)
    {
        currXPoints.push_back(point.x);
    }
    std::sort(currXPoints.begin(), currXPoints.end());

    // Get the second element in the sorted array to skip possible outliers
    const double prevMinX = prevXPoints[1];
    const double currMinX = currXPoints[1];

    // calculate time to collision
    const double dT = 1.0 / frameRate;
    TTC = currMinX * dT / (prevMinX - currMinX);
}
```



### FP.3 Associate Keypoint Correspondences with Bounding Boxes

> Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

There are two main steps needed to associate the keypoints with the region of interest.

The first is checking if the keypoint is located within the bounding box. This can be done using the `contains()` function available in the `cv::Rect` type.

The second is to remove the mismatches. Most of keypoints are expected to have similar variations in distance from one image to the next because the car is a rigid object. Therefore, it is possible to filter the outliers by computing the average distance variation and removing the ones that are too far away from this value.

For the given data, since the variation between each frame is small, a threshold of 3 was used to discard the outliers discarded. The implementation can be seen below:

```cpp
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    //Create vector with all matches inside the ROI
    std::vector<cv::DMatch> kptMatchesInROI;
    for (const cv::DMatch& match : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            kptMatchesInROI.push_back(match);
        }
    }

    //Create vector with distances between the matches inside ROI
    std::vector<double> kptDistances;
    double averageDist = 0;
    for (const cv::DMatch& match : kptMatchesInROI)
    {
        double dist = cv::norm(kptsPrev[match.queryIdx].pt - kptsCurr[match.trainIdx].pt);
        kptDistances.push_back(dist);
        averageDist += dist;
    }
    averageDist = averageDist/kptDistances.size();

    // Only take into account the matches that are not too far from the average
    for (int i=0; i<kptDistances.size(); i++)
    {
        if (kptDistances[i] < averageDist+3)
        {
            boundingBox.kptMatches.push_back(kptMatchesInROI[i]);
        }
    }
}
```



### FP.4 Compute Camera-based TTC

> Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

By understanding the pinhole model for the camera, shown in the image below (from Udacity), it is possible to calculate the TTC by observing the ratio between the images project in the image plane (h1 and h2).

![][img_udacity_camerattc1]

The distance is computed by the combination of keypoint matches from the previous and current images.
To get a stable value for the size variation of the preceding vehicle, several distance measurements are used by computing all combinations of the keypoints. This idea can be easily visualized in the image below from Udacity.


![][img_udacity_camerattc2]

All the average distances are stored in an auxiliary vector and to get a robust result, the median value is calculated. The implementation below is based on the algorithms from the lecture.

```cpp
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}
```

### FP.5 Performance Evaluation 1

> Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

To have an idea of the estimated TTC it is possible to calculate it manually using the top view perspective of the Lidar points.
The table below shows the results for the first three iterations considering the car has a constant speed between each data frame.

![][img_lidar_table]

The actual result for the TCC obtained using the implemented code is shown in the plot below.

![][plot_lidar]

Looking at the graph, some measurements do not seem plausible especially two:
1. Lidar-based TTC from frames 5 to 6
2. Lidar-based TTC from frames 6 to 7

In both cases, what causes the error in the TTC is the variation in the measured distance.

Inspecting the top-view perspective, it is not easy to identify an outlier among the lidar points. Especially considering that they are filtered in the algorithm.

Frame 05
![][img_topv1]
Frame 06
![][img_topv2]
Frame 07
![][img_topv3]

It is more likely that the Lidar beams are reflecting from different parts of the car, like a bumper for example, and it provides a different distance estimation. This issue can be visualized in the bottom of the bounding box of two images below.

![][img_bump1]
![][img_bump2]


### FP.6 Performance Evaluation 2

> Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Several combinations of keypoint detectors and descriptors are possible.
In order to facilitate the evaluation of the results, the output was customized to have a CSV format. To enable this type of output, the `csvOutput` boolean flag was created. The output can be redirected to a file and evaluated using the [PerformanceEvaluation.ipynb](./PerformanceEvaluation.ipynb) Jupyter Notebook.

The table below shows all possible combinations that were evaluated

![][table_all_combinations]

To have an overall idea of the performance, the number of keypoint matches and the detection time is summarized in the table below. Each value represents the average of each Keypoint Detector combined with all possible Descriptors.

![][plot_detection_overview]


The camera-based TTC varies a lot depending on the selected algorithm. The plot below has all possible combinations and it is quite cluttered making it hard to understand which ones are performing well.

![][plot_all_combinations]

Analyzing each group of detector individually, it is possible to observe that some algorithms are very unstable.
HARRIS and ORB, for instance, have several gaps that are caused by the lack of sufficient keypoint matches between the frames. As shown in the "Performance of Keypoint Detectors" plot above, they are the detectors with the lowest amount of keypoints.

![][plot_harris]
![][plot_orb]
![][plot_brisk]


Some other algorithms, such as AKAZE, FAST and SIFT. provide results that are much more stable and therefore more appropriate for camera-based TTC estimation.

![][plot_akaze]
![][plot_fast]
![][plot_sift]


For this test, the SHITOMASI Detector had the most stable results. The computed TTC mainly decreasing over time, which is exactly what is expected from an approaching vehicle.

![][plot_shitomasi]
