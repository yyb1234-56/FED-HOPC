#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/features2d.hpp>
#include "LibHeader.h"
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;


Mat MotionEstimation(IplImage *cvlastimg, IplImage *cvcurimg,
	cv::Ptr<cv::FeatureDetector> MatchDetector, double Efficient2reliable, int mode);

Mat estimateGlobalMotionHomographyRansac( vector<Point2f> &points0, vector<Point2f> &points1,
	float *rmse, int *ninliers);

void KeyPointSelect(cv::KeyPoint* KeyPoint, float w, float h, int gw, CvPoint2D32f* cornersA, float * presponses);

Mat estimateGlobalMotionAffineRobust(
	vector<Point2f> &points0, vector<Point2f> &points1,
	RansacParams &params, float *rmse, int *ninliers, int mode);

static Mat estimateGlobMotionLeastSquaresAffine(int npoints, Point2f *points0, Point2f *points1, float *rmse);

inline float sqr(float x) { return x * x; }

static void MyErrorExit(const char * message)
{

	fprintf(stderr, message);
	getchar();
	exit(1);
} 

int main()
{

	Mat H;
	
	IplImage *cvlastimg = cvLoadImage("2.jpg");
	IplImage *cvcurimg  = cvLoadImage("4.jpg");

	cv::Ptr<cv::FeatureDetector> MatchDetector = FastFeatureDetector::create(FAST_DETECT_THRESHOLD);

	double begin_time = getTickCount();

	H = MotionEstimation(cvlastimg, cvcurimg, MatchDetector, 0.1, 2);

	double time = (getTickCount() - begin_time)/getTickFrequency();
	cout << time << endl;
	cout <<H.channels() << endl;

	Mat src = cvarrToMat(cvlastimg);
	Mat One_mat(src.size(),CV_8UC1, Scalar(1));
	Mat dst, one_dst;
	warpPerspective(src, dst, H, src.size());
	warpPerspective(One_mat, one_dst, H, One_mat.size());
	Mat Fusion_img(src.size(), CV_8UC3, Scalar(0));
	Fusion_img = 0.5*dst + cvarrToMat(cvcurimg);
	//for (int i=0;i<one_dst.rows;i++)
	//{
	//	for (int j = 0;j<one_dst.cols;j++)
	//	{
	//		one_dst.at<uchar>(i,j)>0? Fusion_img.at<Vec3i>(i,j) = 0.5*dst;
	//	}
	//}
	
	imshow("±ä»»ºó", dst);
	cvShowImage("ÓÒÍ¼",cvcurimg);
	imshow("ÈÚºÏÍ¼Ïñ", Fusion_img);

	waitKey();
	return 0;
}

Mat MotionEstimation(IplImage *cvlastimg, IplImage *cvcurimg,
	cv::Ptr<cv::FeatureDetector> MatchDetector, double Efficient2reliable, int mode)
{

	IplImage *cvlastimgGray = cvCreateImage(cvGetSize(cvlastimg), IPL_DEPTH_8U, 1);
	cvlastimg->nChannels == 3 ? cvCvtColor(cvlastimg, cvlastimgGray, CV_RGB2GRAY) : cvCopy(cvlastimg, cvlastimgGray);

	IplImage *cvcurimgGray = cvCreateImage(cvGetSize(cvcurimg), IPL_DEPTH_8U, 1);
	cvcurimg->nChannels == 3 ? cvCvtColor(cvcurimg, cvcurimgGray, CV_RGB2GRAY) : cvCopy(cvcurimg, cvcurimgGray);

	CvSize      img_sz = cvGetSize(cvlastimgGray);
	int         win_size = cvRound(WIN_WIDTH_LK + Efficient2reliable * WIN_WIDTH_LK_R);
	int              corner_count = MAX_CORNERS;
	CvPoint2D32f* cornersA = new CvPoint2D32f[MAX_CORNERS];
	CvPoint2D32f* cornersB = new CvPoint2D32f[MAX_CORNERS];
	CvPoint2D32f* grids = new CvPoint2D32f[MAX_CORNERS];
	CvPoint2D32f* gride = new CvPoint2D32f[MAX_CORNERS];

	char* features_found = new char[MAX_CORNERS];
	float* feature_errors = new float[MAX_CORNERS];

	CvSize pyr_sz = cvSize(cvlastimgGray->width + 8, cvcurimgGray->height / 3);
	IplImage* pyrA = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);
	IplImage* pyrB = cvCreateImage(pyr_sz, IPL_DEPTH_32F, 1);

	//feature point detection
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat last = cv::cvarrToMat(cvlastimgGray);
	MatchDetector->detect(last, keypoints);

	int gw, gh, i;
	float w, h;
	gw = cvRound(GRID_ROW + GRID_ROW_R * Efficient2reliable);  w = (float)cvlastimg->width / (float)(gw);
	gh = cvRound(GRID_COL + GRID_COL_R * Efficient2reliable);  h = (float)cvlastimg->height / (float)(gh);

	for (i = 0; i<gw*gh; i++) cornersA[i].x = cornersA[i].y = -1;
	float * presponses = (float *)malloc(sizeof(float) * gw * gh);
	for (i = 0; i<gw * gh; i++) presponses[i] = MIN_VALUE;
	if (presponses == NULL) MyErrorExit("MotionEstimation_FASTKLTNCC::presponses malloc error.");

	//select the key points
	for (i = 0; i<(int)keypoints.size(); i++)
		KeyPointSelect(&keypoints[i], w, h, gw, cornersA, presponses);

	for (i = 0; i<gw*gh; i++) {
		if (cornersA[i].x == -1 || cornersA[i].y == -1) {
			cornersA[i].x = (float)((i % gw) * (cvlastimgGray->width - 2 * BORDER_DISTANCE) / (float)(gw - 1) + BORDER_DISTANCE);
			cornersA[i].y = (float)((i / gh) * (cvlastimgGray->height - 2 * BORDER_DISTANCE) / (float)(gh - 1) + BORDER_DISTANCE);
		}
	}
	corner_count = i;

	//KLT track
	cvFindCornerSubPix(cvlastimgGray, cornersA, corner_count, cvSize(win_size, win_size), cvSize(-1, -1), \
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.1));

	cvCalcOpticalFlowPyrLK(cvlastimgGray, cvcurimgGray, \
		pyrA, pyrB, cornersA, cornersB, corner_count, cvSize(win_size, win_size), 5, features_found, feature_errors, \
		cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.1), 0);

	cv::Point2f pt;
	std::vector<cv::Point2f> srcps, dstps;
	int gridn = corner_count = 0; int ngw = cvRound(NCC_GW + Efficient2reliable * NCC_GW_R);

	for (int i = 0; i < ngw * ngw; i++)
	{
		grids[i].x = (float)((i % ngw) * (cvlastimgGray->width - 2 * BORDER_DISTANCE) / (float)(ngw - 1) + ngw);
		grids[i].y = (float)((i / ngw) * (cvlastimgGray->height - 2 * BORDER_DISTANCE) / (float)(ngw - 1) + ngw);
	}
	w = cvlastimgGray->width / (float)ngw; 	h = cvlastimgGray->height / (float)ngw;

	//re-track the key point if it's not found or its feature_errors is too big
	for (i = 0; i< gw * gh && i < MAX_CORNERS; i++) {
		if (features_found[i] && feature_errors[i] < MAX_ERRORS)
		{
			pt.x = cornersA[i].x; pt.y = cornersA[i].y; srcps.push_back(pt);
			pt.x = cornersB[i].x; pt.y = cornersB[i].y; dstps.push_back(pt);
			int idx = (int)(pt.x / w); if (idx >= ngw) idx = ngw - 1;
			int idy = (int)(pt.y / h); if (idy >= ngw) idy = ngw - 1;
			grids[idx + idy * ngw].x = -1;  	grids[idx + idy * ngw].y = -1;
		}
	}

	for (int i = 0; i < ngw * ngw; i++)
	{
		if (grids[i].x >= 0 && grids[i].y >= 0)
		{
			grids[gridn++] = grids[i];
		}
	}

	//use NCC template match to perform re-track
	IplImage *move, *base;
	int mw = cvRound(MOVE_WIDTH + MOVE_WIDTH_R * Efficient2reliable);
	int bw = cvRound(BASE_WIDTH + BASE_WIDTH_R * Efficient2reliable);
	move = cvCreateImage(cvSize(mw, mw), cvlastimgGray->depth, cvlastimgGray->nChannels);
	base = cvCreateImage(cvSize(bw, bw), cvlastimgGray->depth, cvlastimgGray->nChannels);
	int iw = base->width - move->width + 1;
	int ih = base->height - move->height + 1;
	IplImage * cvresults = cvCreateImage(cvSize(iw, ih), 32, 1);
	double maxv; CvPoint maxp; IplImage *below1, *up1, *below2, *up2;
	up1 = NULL; up2 = NULL;

	int PYRAMID_LEVEL = 3;
	if (cvlastimg->width + cvlastimg->height >= 1200) PYRAMID_LEVEL = 4;
	if (cvlastimg->width + cvlastimg->height >= 2400) PYRAMID_LEVEL = 5;

	IplImage ** lastimgs = (IplImage **)malloc(sizeof(IplImage *) * PYRAMID_LEVEL);
	IplImage ** curimgs = (IplImage **)malloc(sizeof(IplImage *) * PYRAMID_LEVEL);

	for (i = 0; i < PYRAMID_LEVEL; i++) 
	{
		if (i == 0) {
			up1 = below1 = cvlastimgGray; 
			up2 = below2 = cvcurimgGray;
		}
		else {
			below1 = up1; 
			below2 = up2;
			up1 = cvCreateImage(cvSize(below1->width / 2, below1->height / 2), below1->depth, below1->nChannels);
			up2 = cvCreateImage(cvSize(below2->width / 2, below2->height / 2), below2->depth, below2->nChannels);
			cvResize(below1, up1); 
			cvResize(below2, up2);
		}
		lastimgs[i] = up1; curimgs[i] = up2;
	}

	for (i = 0; i < PYRAMID_LEVEL; i++) {
		int index = PYRAMID_LEVEL - i - 1;
		double dx, dy, tx, ty;
		IplImage * last = lastimgs[index];
		IplImage * cur = curimgs[index];

		for (int g = 0; g < gridn; g++) {
			tx = grids[g].x / pow((double)2, (double)index);
			ty = grids[g].y / pow((double)2, (double)index);
			if (i == 0) {
				dx = tx; dy = ty;
			}
			else {
				dx = gride[g].x * 2.0; dy = gride[g].y * 2.0;
			}
			int sx = cvRound(tx) - move->width / 2;
			int sy = cvRound(ty) - move->height / 2;
			int difx, dify;	difx = dify = 0;
			if (sx < 0) {
				difx = sx; sx = 0;
			}
			if (sy < 0) {
				dify = sy; sy = 0;
			}
			if (sx + move->width > last->width) {
				difx = sx + move->width - last->width;
				sx = last->width - move->width;
			}
			if (sy + move->height > last->height) {
				dify = sy + move->height - last->height;
				sy = last->height - move->height;
			}
			cvSetImageROI(last, cvRect(sx, sy, move->width, move->height));
			cvCopy(last, move);
			cvResetImageROI(last);

			int bx = cvRound(dx) - base->width / 2;
			int by = cvRound(dy) - base->height / 2;
			if (bx < 0) bx = 0; if (by < 0) by = 0;
			if (bx + base->width > cur->width) bx = cur->width - base->width;
			if (by + base->height > cur->height) by = cur->height - base->height;
			cvSetImageROI(cur, cvRect(bx, by, base->width, base->height));
			cvCopy(cur, base); 
			cvResetImageROI(cur);

			cvMatchTemplate(base, move, cvresults, CV_TM_CCOEFF_NORMED);
			cvMinMaxLoc(cvresults, NULL, &maxv, NULL, &maxp);
			gride[g].x = float(difx + bx + maxp.x + move->width / 2);
			gride[g].y = float(dify + by + maxp.y + move->height / 2);
		}
	}

	for (i = 1; i < PYRAMID_LEVEL; i++) {
		cvReleaseImage(lastimgs + i);
		cvReleaseImage(curimgs + i);
	}
	free(lastimgs);  free(curimgs);
	cvReleaseImage(&move); cvReleaseImage(&base); cvReleaseImage(&cvresults);
	for (i = 0; i<gridn; i++) {
		pt.x = grids[i].x; pt.y = grids[i].y; srcps.push_back(pt);
		pt.x = gride[i].x; pt.y = gride[i].y; dstps.push_back(pt);
	}

	// estimate global motion using RANSAC
	int ninliers; float rmse; Mat M;
	float rmse_threshold = (cvlastimg->width + cvlastimg->height) * MAX_RMSE_RATIO / 2.0f;
	if (mode == HOMOGRAPHY)
	{
		M = estimateGlobalMotionHomographyRansac(
			srcps, dstps, &rmse, &ninliers);
	}
	else
	{
		RansacParams RPs = (RansacParams::affine2dMotionStd());
		M = estimateGlobalMotionAffineRobust(
			srcps, dstps, RPs, &rmse, &ninliers, mode);
	}

	//return identity matrix if the inlier ratio is too small or the inlier rmse is too big.
	if (rmse > rmse_threshold || (float)(ninliers) / (float)srcps.size() < MIN_INLIERS)
	{
		M = Mat::eye(3, 3, CV_32F);
		fprintf(stderr, "estimate error\n");
	}

	keypoints.clear();
	cvReleaseImage(&pyrA); 	cvReleaseImage(&pyrB);
	delete[]cornersB; 	delete[]cornersA; delete[]gride; delete[]grids;
	delete[]features_found; delete[]feature_errors;
	free(presponses); cvReleaseImage(&cvcurimgGray); cvReleaseImage(&cvlastimgGray);
	return M;
}

Mat estimateGlobalMotionHomographyRansac( vector<Point2f> &points0, vector<Point2f> &points1, float *rmse, int *ninliers)
{
	Mat_<float> H;
	std::vector <unsigned char> inMask(points0.size()); int innum = 0;
	H = findHomography(points0, points1, CV_RANSAC, 3, inMask);
	std::vector<cv::Point2f>ps0, ps1, ps2;
	for (int i = 0; i < inMask.size(); i++)
	{
		if (inMask[i])
		{
			ps0.push_back(points0[i]);
			ps1.push_back(points1[i]);
		}
	}
	(*ninliers) = (int)(ps0.size());
	perspectiveTransform(ps0, ps2, H);
	*rmse = static_cast<float>(norm(ps1, ps2, NORM_L2) / sqrt(static_cast<double>(*ninliers)));
	return H;
}

void KeyPointSelect(cv::KeyPoint* KeyPoint, float w, float h, int gw, CvPoint2D32f* cornersA, float * presponses)
{
	int x, y;
	x = int(KeyPoint->pt.x / w); y = int(KeyPoint->pt.y / h);
	if (cornersA[x + y*gw].x == -1 || cornersA[x + y*gw].y == -1 || presponses[x + y*gw] < KeyPoint->response)
	{
		cornersA[x + y*gw].x = KeyPoint->pt.x;
		cornersA[x + y*gw].y = KeyPoint->pt.y;
		presponses[x + y*gw] = KeyPoint->response;
	}
}

Mat estimateGlobalMotionAffineRobust(
	vector<Point2f> &points0, vector<Point2f> &points1,
	RansacParams &params, float *rmse, int *ninliers, int mode)
{
	if (mode != AFFINE)
		MyErrorExit("estimateGlobalMotionAffineRobust:: mode must be AFFINE.");

	CV_Assert(points0.size() == points1.size());

	typedef Mat(*Impl)(int, Point2f*, Point2f*, float*);
	static Impl impls[] = { estimateGlobMotionLeastSquaresAffine };

	const int npoints = static_cast<int>(points0.size());
	if (npoints < params.size)
		return Mat::eye(3, 3, CV_32F);

	const int niters = static_cast<int>(ceil(log(1 - params.prob) /
		log(1 - pow(1 - params.eps, params.size))));

	RNG rng(0);
	vector<int> indices(params.size);
	vector<Point2f> subset0(params.size), subset1(params.size);
	vector<Point2f> subset0best(params.size), subset1best(params.size);
	Mat_<float> bestM;
	int ninliersMax = -1;
	Point2f p0, p1;
	float x, y;

	for (int iter = 0; iter < niters; ++iter)
	{
		for (int i = 0; i < params.size; ++i)
		{
			bool ok = false;
			while (!ok)
			{
				ok = true;
				indices[i] = static_cast<unsigned>(rng) % npoints;
				for (int j = 0; j < i; ++j)
					if (indices[i] == indices[j])
					{
						ok = false; break;
					}
			}
		}
		for (int i = 0; i < params.size; ++i)
		{
			subset0[i] = points0[indices[i]];
			subset1[i] = points1[indices[i]];
		}

		Mat_<float> M = impls[mode - 1](params.size, &subset0[0], &subset1[0], 0);

		int _ninliers = 0;
		for (int i = 0; i < npoints; ++i)
		{
			p0 = points0[i]; p1 = points1[i];
			x = M(0, 0)*p0.x + M(0, 1)*p0.y + M(0, 2);
			y = M(1, 0)*p0.x + M(1, 1)*p0.y + M(1, 2);
			if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
				_ninliers++;
		}
		if (_ninliers >= ninliersMax)
		{
			bestM = M;
			ninliersMax = _ninliers;
			subset0best.swap(subset0);
			subset1best.swap(subset1);
		}
	}

	if (ninliersMax < params.size)
		// compute rmse
		bestM = impls[mode - 1](params.size, &subset0best[0], &subset1best[0], rmse);
	else
	{
		subset0.resize(ninliersMax);
		subset1.resize(ninliersMax);
		for (int i = 0, j = 0; i < npoints; ++i)
		{
			p0 = points0[i]; p1 = points1[i];
			x = bestM(0, 0)*p0.x + bestM(0, 1)*p0.y + bestM(0, 2);
			y = bestM(1, 0)*p0.x + bestM(1, 1)*p0.y + bestM(1, 2);
			if (sqr(x - p1.x) + sqr(y - p1.y) < params.thresh * params.thresh)
			{
				subset0[j] = p0;
				subset1[j] = p1;
				j++;
			}
		}
		bestM = impls[mode - 1](ninliersMax, &subset0[0], &subset1[0], rmse);
	}

	if (ninliers)
		*ninliers = ninliersMax;
	return bestM;
}

static Mat estimateGlobMotionLeastSquaresAffine(int npoints, Point2f *points0, Point2f *points1, float *rmse)
{
	Mat_<float> A(2 * npoints, 6), b(2 * npoints, 1);
	float *a0, *a1;
	Point2f p0, p1;

	for (int i = 0; i < npoints; ++i)
	{
		a0 = A[2 * i];
		a1 = A[2 * i + 1];
		p0 = points0[i];
		p1 = points1[i];
		a0[0] = p0.x; a0[1] = p0.y; a0[2] = 1; a0[3] = a0[4] = a0[5] = 0;
		a1[0] = a1[1] = a1[2] = 0; a1[3] = p0.x; a1[4] = p0.y; a1[5] = 1;
		b(2 * i, 0) = p1.x;
		b(2 * i + 1, 0) = p1.y;
	}

	Mat_<float> sol;
	solve(A, b, sol, DECOMP_SVD);

	if (rmse)
		*rmse = static_cast<float>(norm(A*sol, b, NORM_L2) / sqrt(static_cast<double>(npoints)));

	Mat_<float> M = Mat::eye(3, 3, CV_32F);
	for (int i = 0, k = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j, ++k)
			M(i, j) = sol(k, 0);

	return M;
}