#include <iostream>
#include <math.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <limits>
#include "mpi.h"
using namespace std;
using namespace cv;


float findDistance(int cr,int cb,int cg,int r,int b,int g){
    float distance = (float)sqrt(pow(cr-r,2)+pow(cb-b,2)+pow(cg-g,2));
    return distance;
}

void kmeans(int* pointMat, int* centroids,int*labels, int dataSize, int clusterNum){
    bool flag = false;
    int maxIterations = 1000;
    int iterCount = 0;
    int distance;
    int rank, commsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    int low = (dataSize*rank)/commsize;
    int high = floor((dataSize*(rank+1))/commsize);
    int partSize = high-low;
    int clusterCounts[clusterNum];
    while(iterCount < maxIterations && flag == false){
        int newCentroids[clusterNum*3];
        for(int i = 0;i <clusterNum;i++){
            newCentroids[i*3] = 0;
            newCentroids[i*3+1] = 0;
            newCentroids[i*3+2] = 0;
            clusterCounts[i] = 0;
        }
        for(int i = 0; i < partSize; i++){
            int minDist = std::numeric_limits<int>::max();
            int clusterToAssign = 0;
            for(int j = 0; j < clusterNum; j++){
                distance = findDistance(pointMat[i*3], pointMat[i*3+1], pointMat[i*3+2],
                                        centroids[j*3],centroids[(j*3)+1],centroids[(j*3)+2]);
                if(distance < minDist){
                    minDist = distance;
                    clusterToAssign = j;
                }
            }
            labels[low+i] = clusterToAssign;
            newCentroids[clusterToAssign*3] += pointMat[i*3];
            newCentroids[clusterToAssign*3+1] += pointMat[i*3+1];
            newCentroids[clusterToAssign*3+2] += pointMat[i*3+2];
            clusterCounts[clusterToAssign]++;
        }
        int globalNewCentroids[clusterNum*3];
        int globalClusterCounts[clusterNum];
        MPI_Reduce(newCentroids,globalNewCentroids, clusterNum*3, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(clusterCounts, globalClusterCounts,clusterNum, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        float d = 0;
        if(rank == 0){
            for(int i = 0; i < clusterNum; i++){
                if(globalClusterCounts[i] == 0 ){
                    globalClusterCounts[i] = 1;
                }
                globalNewCentroids[i*3] /= globalClusterCounts[i];
                globalNewCentroids[i*3+1] /= globalClusterCounts[i];
                globalNewCentroids[i*3+2] /= globalClusterCounts[i];
            }

            for(int i = 0; i<clusterNum;i++){
                d += findDistance(centroids[(i*3)],centroids[(i*3)+1],centroids[(i*3)+2],
                                globalNewCentroids[i*3],globalNewCentroids[i*3+1],globalNewCentroids[i*3+2]);
                centroids[(i*3)] = globalNewCentroids[i*3+0];
                centroids[(i*3)+1] = globalNewCentroids[i*3+1];
                centroids[(i*3)+2] = globalNewCentroids[i*3+2];
            }
        }
        MPI_Bcast(centroids, clusterNum *3, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
        iterCount++;
        if(d < 0.001){ flag = true;}
    }
}

int main( int argc, char** argv ){
    if( argc != 3)
    {
     cout <<" Provide image and cluster number" << endl;
     return -1;
    }

    int rank, commsize;
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); //rank of current processor
    MPI_Comm_size(MPI_COMM_WORLD, &commsize); //amount of all processors

    int dataSize, row_size;
    Mat pointMat;
    int clusterNum = atoi(argv[2]);
    int *imagePart, *centroids;
    double start;
    if(rank ==0){
        Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

        if(! image.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            return -1;
        }
        row_size = (int)image.rows;
        pointMat = image.reshape(3, 1).t();
        start = MPI_Wtime();

        dataSize = pointMat.rows*pointMat.cols;
        Scalar r,g,b;
        for(int i=1;i<commsize;i++){
            MPI_Send(&dataSize,1,MPI_INT,i,0,MPI_COMM_WORLD);
            int low = ((dataSize*i)/commsize);
            int high = floor((dataSize*(i+1))/commsize);
            int partSize = high-low;
            int temp[partSize*3];
            for(int j=0;j<partSize;j++){
                r = (int)pointMat.at<uchar>(low+j, 0);
                g = (int)pointMat.at<uchar>(low+j, 1);
                b = (int)pointMat.at<uchar>(low+j, 2);
                temp[(j*3)] = (int)r[0];
                temp[(j*3)+1] = (int)g[0];
                temp[(j*3)+2] = (int)b[0];
            }
            MPI_Send(temp, partSize*3, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        int low = ((dataSize*rank)/commsize);
        int high = floor((dataSize*(rank+1))/commsize);
        int partSize = high-low;
        imagePart = (int *)malloc(partSize*3*sizeof(int));
        for(int i=0;i<partSize;i++){
            r = (int)pointMat.at<uchar>(low+i, 0);
            g = (int)pointMat.at<uchar>(low+i, 1);
            b = (int)pointMat.at<uchar>(low+i, 2);
            imagePart[i*3] = (int)r[0];
            imagePart[i*3+1] = (int)g[0];
            imagePart[i*3+2] = (int)b[0];
        }

        centroids = (int *)malloc(clusterNum*3*sizeof(int));

        int x;
        srand((unsigned)time(0));
        for(int i = 0; i< clusterNum; i++){
            x = rand() % (int)pointMat.rows;
            r = (int)pointMat.at<uchar>(x, 0);
            g = (int)pointMat.at<uchar>(x, 1);
            b = (int)pointMat.at<uchar>(x, 2);
            centroids[(i*3)] = (int)r[0];
            centroids[(i*3)+1] = (int)g[0];
            centroids[(i*3)+2] = (int)b[0];
        }

    }else{
        MPI_Recv(&dataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        int low = ((dataSize*rank)/commsize);
        int high = floor((dataSize*(rank+1))/commsize);
        int partSize = high-low;
        imagePart = (int *)malloc(partSize*3*sizeof(int));
        MPI_Recv(imagePart, partSize*3, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        centroids = (int *)malloc(clusterNum*3*sizeof(int));
    }
    MPI_Bcast(centroids, clusterNum *3, MPI_INT, 0, MPI_COMM_WORLD);

    int labels[dataSize];
    for(int i = 0;i<dataSize; i++){
        labels[i] = 0;
    }
    kmeans(imagePart,centroids,labels,dataSize,clusterNum);

    int labels_final[dataSize];
    MPI_Reduce(labels, labels_final, dataSize, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank ==0){
        cout << "PAINT IT BLACK"<<endl;
        for(int i = 0; i< dataSize; i++){
                pointMat.at<uchar>(i, 0) = (int)centroids[labels_final[i]*3];
                pointMat.at<uchar>(i, 1) = (int)centroids[labels_final[i]*3+1];
                pointMat.at<uchar>(i, 2) = (int)centroids[labels_final[i]*3+2];
        }
        double end = MPI_Wtime();
        cout<< "Program took " << end-start<< " seconds" << endl;
        Mat clustered_image = pointMat.reshape(3,row_size);
        //imwrite( "/home/dogu/Desktop/kmeans/Kmeans_OpenCV_MPI/result30.jpg", clustered_image );
        namedWindow( "Display window", WINDOW_AUTOSIZE );
        imshow("Display window", clustered_image);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    waitKey(0);

    MPI_Finalize();
    return 0;
}

