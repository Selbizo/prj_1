#include "wienerFilter.h"

using namespace cv;
using namespace std;

// NO CUDA begins
void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(0, cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];

	Mat outputImg_norm;
	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
	cv::imshow("PSF", outputImg_norm);
}

void calcPSF_circle(Mat& outputImg, Size filterSize, int len, double theta)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(cvRound(double(len) / 2.0), cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];


	Mat outputImg_norm;
	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
	cv::imshow("PSF", outputImg_norm);
}

void fftshift(const Mat& inputImg, Mat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));
	Mat q1(outputImg, Rect(cx, 0, cx, cy));
	Mat q2(outputImg, Rect(0, cy, cx, cy));
	Mat q3(outputImg, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
	Mat planes[2] = { Mat_<double>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_SCALE);

	Mat planesH[2] = { Mat_<double>(H.clone()), Mat::zeros(H.size(), CV_32F) };
	Mat complexH;
	merge(planesH, 2, complexH);
	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0);

	idft(complexIH, complexIH);
	split(complexIH, planes);
	outputImg = planes[0];
}

void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
	Mat h_PSF_shifted;
	fftshift(input_h_PSF, h_PSF_shifted);
	Mat planes[2] = { Mat_<double>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	Mat denom;
	pow(abs(planes[0]), 2, denom);
	denom += nsr;
	divide(planes[0], denom, output_G);
}

void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
{
	int Nx = inputImg.cols;
	int Ny = inputImg.rows;
	Mat w1(1, Nx, CV_32F, Scalar(0));
	Mat w2(Ny, 1, CV_32F, Scalar(0));

	double* p1 = w1.ptr<double>(0);
	double* p2 = w2.ptr<double>(0);
	double dx = double(2.0 * CV_PI / Nx);
	double x = double(-CV_PI);
	for (int i = 0; i < Nx; i++)
	{
		p1[i] = double(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
		x += dx;
	}
	double dy = double(2.0 * CV_PI / Ny);
	double y = double(-CV_PI);
	for (int i = 0; i < Ny; i++)
	{
		p2[i] = double(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
		y += dy;
	}
	Mat w = w2 * w1;
	multiply(inputImg, w, outputImg);
}

//NO CUDA end

//WITH CUDA begins

void GcalcPSF(cuda::GpuMat& outputImg, Size filterSize, Size psfSize, double len, double theta)
{
	// ������� GpuMat ��� ���������� ��������
	int scale = 8;
	cuda::GpuMat h(filterSize, CV_32F, Scalar(0));
	Mat hCpu(psfSize, CV_32F, Scalar(0));
	Mat hCpuBig(Size(psfSize.width * scale, psfSize.height * scale), CV_32F, Scalar(0));
	// ����� �������
	Point center(psfSize.width * scale / 2, psfSize.height * scale / 2);

	// ������� �������
	Size axes(scale, cvRound(double(len * scale + scale) / 2.0f));
	Size axes2(scale, cvRound(double(len * scale + scale) / 4.0f));
	Size axes3(scale, cvRound(double(len * scale + scale) / 6.0f));
	// ���� �������� �������
	double angle = 90.0 - theta;

	// ������ ������ �� GpuMat

	ellipse(hCpuBig, center, axes, angle, 0, 360, Scalar(0.2), FILLED);

	ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.4), FILLED);
	ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.9), FILLED);
	resize(hCpuBig, hCpu, psfSize, INTER_LINEAR);
	if (hCpu.cols > h.cols / 2)
		resize(hCpu, hCpu, Size(h.cols / 2 - 1, hCpu.rows), INTER_LINEAR);
	if (hCpu.rows > h.rows / 2)
		resize(hCpu, hCpu, Size(hCpu.cols, h.rows / 2 - 1), INTER_LINEAR);
	// 
		// �������� ����� ��������� ����� � ������� ����
	imshow("PSF Cpu", hCpu);
	//hCpu(Rect(0, 0, psfSize.width, psfSize.height)).copyTo(h(Rect((filterSize.width - psfSize.width) / 2, (filterSize.height - psfSize.height) / 2, psfSize.width, psfSize.height)));
	hCpu(Rect(0, 0, hCpu.cols, hCpu.rows)).copyTo(h(Rect((filterSize.width - hCpu.cols) / 2, (filterSize.height - hCpu.rows) / 2, hCpu.cols, hCpu.rows)));

	//h.upload(hCpu);



	// ��������� ��� �������� GpuMat
	Scalar summa = cuda::sum(h);

	// ����� GpuMat �� �����
	cuda::divide(h, Scalar(summa[0]), outputImg);


}

void GcalcPSFCircle(cuda::GpuMat& outputImg, Size filterSize, double len, double theta)
{
	// ������� GpuMat ��� ���������� ��������
	cuda::GpuMat h(filterSize, CV_32F, Scalar(0));
	Mat hCpu(filterSize, CV_32F, Scalar(0));
	// ����� �������
	Point center(filterSize.width / 2, filterSize.height / 2);

	// ������� �������
	Size axes(cvRound(double(len) / 2.0), cvRound(double(len) / 2.0));
	Size axes2(0, cvRound(double(len) / 4.0f));
	// ���� �������� �������
	double angle = 90.0 - theta;

	// ������ ������ �� GpuMat

	ellipse(hCpu, center, axes, angle, 0, 360, Scalar(255), FILLED);
	//ellipse(hCpu, center, axes2, angle, 0, 360, Scalar(255), FILLED);
	blur(hCpu, hCpu, Size(5, 5));
	h.upload(hCpu);

	// ��������� ��� �������� GpuMat
	Scalar summa = cuda::sum(h);

	// ����� GpuMat �� �����
	cuda::divide(h, Scalar(summa[0]), outputImg);
}

void Gfftshift(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	cuda::GpuMat q0(outputImg, Rect(0, 0, cx, cy));
	cuda::GpuMat q1(outputImg, Rect(cx, 0, cx, cy));
	cuda::GpuMat q2(outputImg, Rect(0, cy, cx, cy));
	cuda::GpuMat q3(outputImg, Rect(cx, cy, cx, cy));
	cuda::GpuMat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void Gfilter2DFreq(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& H)
{
	// ��������� ������� �����������
	cuda::GpuMat inputClone;
	inputImg.copyTo(inputClone);

	// ������� GpuMat ��� ������ �����
	cuda::GpuMat zeroMat(inputImg.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� � ����������� �������
	vector<cuda::GpuMat> planes = { inputClone, zeroMat };
	cuda::GpuMat complexInput;
	cuda::merge(planes, complexInput);

	// ������ �������������� �����
	//cuda::dft(complexInput, complexInput, complexInput.size(), DFT_SCALE | DFT_COMPLEX_OUTPUT);
	cuda::dft(complexInput, complexInput, complexInput.size(), DFT_SCALE);

	// ��������� ������
	cuda::GpuMat HClone;
	H.copyTo(HClone);

	// ������� GpuMat ��� ������ ����� �������
	cuda::GpuMat zeroMatH(H.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� ������� � ����������� �������
	vector<cuda::GpuMat> planesH = { HClone, zeroMatH };
	cuda::GpuMat complexH;
	cuda::merge(planesH, complexH);

	// ��������� ��������
	cuda::GpuMat complexOutput;
	cuda::mulSpectrums(complexInput, complexH, complexOutput, 0);

	// �������� �������������� �����
	cuda::dft(complexOutput, complexOutput, complexOutput.size(), DFT_INVERSE);

	// ��������� ����������� ������� �� ��� �����
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexOutput, planesOut);

	// ������ ��������� �������� ����������� ����������
	outputImg = planesOut[0];
}

void Gfilter2DFreqV2(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& complexH, Ptr<cuda::DFT>& forwardDFT, Ptr<cuda::DFT>& inverseDFT)
{
	// ��������� ������� �����������
	//cuda::GpuMat inputClone;
	//inputImg.copyTo(inputClone);

	// ������� GpuMat ��� ������ �����
	cuda::GpuMat zeroMat(inputImg.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� � ����������� �������
	vector<cuda::GpuMat> planes = { inputImg, zeroMat };
	cuda::GpuMat complexInput;
	cuda::merge(planes, complexInput);

	// ������ �������������� �����

	forwardDFT->compute(complexInput, complexInput);
	// ��������� ��������
	cuda::GpuMat complexOutput;
	cuda::mulSpectrums(complexInput, complexH, complexOutput, 0);
	// �������� �������������� �����
	inverseDFT->compute(complexOutput, complexOutput);

	// ��������� ����������� ������� �� ��� �����
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexOutput, planesOut);

	// ������ ��������� �������� ����������� ����������
	outputImg = planesOut[0];
}

void GcalcWnrFilter(const cuda::GpuMat& input_h_PSF, cuda::GpuMat& output_G, double nsr)
{
	// ������� ����� �������� �����������
	//cuda::GpuMat h_PSF_clone;
	//input_h_PSF.copyTo(h_PSF_clone);

	// ��������� ����� �����
	cuda::GpuMat h_PSF_shifted;
	Gfftshift(input_h_PSF, h_PSF_shifted);

	// ������� GpuMat ��� ������ �����
	cuda::GpuMat zeroMat(h_PSF_shifted.size(), CV_32F, Scalar(0));

	// ���������� �������������� � ������ ����� � ����������� �������
	vector<cuda::GpuMat> planes = { h_PSF_shifted, zeroMat };
	cuda::GpuMat complexI;
	cuda::merge(planes, complexI);

	// ������ �������������� �����
	//cuda::dft(complexI, complexI, complexI.size(), DFT_COMPLEX_OUTPUT);
	cuda::dft(complexI, complexI, complexI.size());

	// ��������� ����������� ������� �� ��� �����
	vector<cuda::GpuMat> planesOut;
	cuda::split(complexI, planesOut);

	// ��������� �����������
	cuda::GpuMat denom;
	cuda::magnitude(planesOut[0], planesOut[1], denom);
	cuda::pow(denom, 2, denom);
	//denom += nsr;
	cuda::add(denom, nsr, denom);

	// �������
	cuda::divide(planesOut[0], denom, output_G);
}

void Gedgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
{
	int Nx = inputImg.cols;
	int Ny = inputImg.rows;
	Mat w1(1, Nx, CV_32F, Scalar(0));
	Mat w2(Ny, 1, CV_32F, Scalar(0));

	double* p1 = w1.ptr<double>(0);
	double* p2 = w2.ptr<double>(0);
	double dx = double(2.0 * CV_PI / Nx);
	double x = double(-CV_PI);
	for (int i = 0; i < Nx; i++)
	{
		p1[i] = double(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
		x += dx;
	}
	double dy = double(2.0 * CV_PI / Ny);
	double y = double(-CV_PI);
	for (int i = 0; i < Ny; i++)
	{
		p2[i] = double(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
		y += dy;
	}
	Mat w = w2 * w1;
	multiply(inputImg, w, outputImg);
}

void channelWiener(const cuda::GpuMat* gChannel, cuda::GpuMat* gChannelWiener,
	const cuda::GpuMat* complexH, cv::Ptr<cuda::DFT>* forwardDFT, cv::Ptr<cuda::DFT>* inverseDFT)
{
	Gfilter2DFreqV2(*gChannel, *gChannelWiener, *complexH, *forwardDFT, *inverseDFT);
}
//WITH CUDA ends