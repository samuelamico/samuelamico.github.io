#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define RADIUS 100

using namespace cv;
using namespace std;

// Existe um arquivo, no caso o exercicio anterior onde eu gerei uma imagem
// mal iluminada utilizando a funca gamma.

Mat imaginaryInput, complexImage, multsp;
Mat padded, filter, mag;
Mat image,imagegray, tmp;
Mat_<float> realInput,zeros;
vector<Mat> planos;

float mean;
char key;

int dft_M, dft_N;

char *name; 

// variaveis para filtro
float yl = 0;
int yl_slider = 0;
int yl_slider_max = 100;

float yh = 0;
int yh_slider =50;
int yh_slider_max=100;

float d0 = 0;
int d0_slider=50;
int d0_slider_max=100;

float c=0;
int c_slider=5;
int c_slider_max=100;

char TrackbarName[50];

// funcao de trocar os quadrantes:

void deslocaDFT(Mat& image)
{
	Mat tmp2,A,B,C,D;
	
	// se a imagem tiver tamanho impar, recortaa regiao para evitar copias de tamano desigual
	image = image(Rect(0,0, image.cols & -2, image.rows & -2));
	int cx = image.cols/2;
	int cy = image.rows/2;
	
	//reoganiza os quadrantes
	//AB -> DC
	//CD -> BA
	A = image(Rect(0,0,cx,cy));
	B = image(Rect(cx,0,cx,cy));
	C = image(Rect(0,cy,cx,cy));
	D = image(Rect(cx,cy,cx,cy));
	
	//A <-> D
	A.copyTo(tmp2); D.copyTo(A); tmp2.copyTo(D);
	
	//C <-> B
	C.copyTo(tmp2); B.copyTo(C); tmp2.copyTo(B);
	
}

// funcao do filtro homomorfico
void on_trackbar_homomorphic(int, void*) 
{
    yl = 26; //(float) yl_slider / 100.0;
    yh = 44; //(float) yh_slider / 100.0;
    d0 = 35; //25.0 * d0_slider / 100.0;
    c  = 8; //(float) c_slider  / 100.0;

    cout << "yl = " << yl << endl;
    cout << "yh = " << yh << endl;
    cout << "d0 = " << d0 << endl;
    cout << "c = "  << c  << endl;

    image = imread(name);
    cvtColor(image, imagegray, CV_BGR2GRAY);
    imshow("original", imagegray);

    // realiza o padding da imagem
    copyMakeBorder(imagegray, padded, 0,
                   dft_M - image.rows, 0,
                   dft_N - image.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    // limpa o array de matrizes que vao compor a
    // imagem complexa
    planos.clear();
    // cria a compoente real
    realInput = Mat_<float>(padded);
    // insere as duas componentes no array de matrizes
    planos.push_back(realInput);
    planos.push_back(zeros);

    // combina o array de matrizes em uma unica
    // componente complexa
    merge(planos, complexImage);

    // calcula o dft
    dft(complexImage, complexImage);

    // realiza a troca de quadrantes
    deslocaDFT(complexImage);
    
    // filtro homomorfico
    for(int i=0; i < tmp.rows; i++){
        for(int j=0; j < tmp.cols; j++){
            float d2 = (i-dft_M/2)*(i-dft_M/2)+(j-dft_N/2)*(j-dft_N/2);
            //cout << "d2 = " << d2 << endl;
            tmp.at<float> (i,j) = (yh-yl)*(1.0 - (float)exp(-(c*d2/(d0*d0)))) + yl;
        }
    }

    // cria a matriz com as componentes do filtro e junta
    // ambas em uma matriz multicanal complexa
    Mat comps[]= {tmp, tmp};
    merge(comps, 2, filter);
    
    // aplica o filtro frequencial
    mulSpectrums(complexImage,filter,complexImage,0);

    // troca novamente os quadrantes
    deslocaDFT(complexImage);

    // calcula a DFT inversa
    idft(complexImage, complexImage);

    // limpa o array de planos
    planos.clear();

    // separa as partes real e imaginaria da
    // imagem filtrada
    split(complexImage, planos);

    // normaliza a parte real para exibicao
    normalize(planos[0], planos[0], 0, 1, CV_MINMAX);
    imshow("filtrada", planos[0]);

}



int main(int argc, char** argv)
{
   namedWindow("ponte mal iluminada",WINDOW_NORMAL);
   namedWindow("ponte filtrada",WINDOW_NORMAL);
   
   if (argc != 2)
   {
	  cerr << "erro";
	  return 1;
   }

   name = argv[1];
   image = imread(name);
   
   //Primeiro identificar os melhores valores para calcular a dft mais otimizado
   dft_M = getOptimalDFTSize(image.rows);
   dft_N = getOptimalDFTSize(image.cols);
   
   // A func copyMakeBorder cria uma versao da imagem com uma borda preenchida de zeros
   copyMakeBorder(image , padded , 0,
					dft_M - image.rows, 0,
					dft_N - image.cols,
					BORDER_CONSTANT, Scalar::all(0));
   
   // parte imaginaria da matriz complexa
   zeros = Mat_<float>::zeros(padded.size());
   
   // prepara a matriz complexa para ser preenchida
   complexImage = Mat(padded.size(), CV_32FC2, Scalar(0));
   
   // funcao de transferencia do filtro deve ter o mesmo tamanho e tipo da matriz complexa
   filter = complexImage.clone();
   
   // cria uma matriz temporaria para compententes real e imaginaria do filtro
   tmp = Mat(dft_M , dft_N, CV_32F);
   
   // funcao do filtro esta ali em cima para ficar mais bonito
   
   //Tracks:
    // Inicializar trackbars
    
    /*
    sprintf( TrackbarName, "yl" );
    createTrackbar( TrackbarName, "filtrada",
                    &yl_slider,
                    yl_slider_max,
                    on_trackbar_homomorphic );
    
    sprintf( TrackbarName, "yh" );
    createTrackbar( TrackbarName, "filtrada",
                    &yh_slider,
                    yh_slider_max,
                    on_trackbar_homomorphic );
    
    sprintf( TrackbarName, "d_zero" );
    createTrackbar( TrackbarName, "filtrada",
                    &d0_slider,
                    d0_slider_max,
                    on_trackbar_homomorphic );
    
    sprintf( TrackbarName, "c" );
    createTrackbar( TrackbarName, "filtrada",
                    &c_slider,
                    c_slider_max,
                    on_trackbar_homomorphic );

    on_trackbar_homomorphic(100, NULL);
    */ 

    while (1) {
        key = (char) waitKey(10);
        if( key == 27 ) break;
    }

    return 0;
}
