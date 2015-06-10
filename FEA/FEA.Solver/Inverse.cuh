// Hard Coded inverse for 2x2 3x3 and 4x4 matrices

// Wolfram Alpha input: inverse {{a, b}, {c, d}} 
// Wolfram Alpha input: inverse {{a, b, c}, {d, e, f}, {g, h, i}}
// Wolfram Alpha input: inverse {{a, b, c, d}, {e, f, g, h}, {i, j, k, l}, {m, n, o, p}} 

template <typename T>
__device__ void Inverse2(T A[2][2]) {
	T a = A[0][0]; T b = A[0][1];
	T c = A[1][0]; T d = A[1][1];
	
	T M = 1 / (a*d - b*c);
	
	A[0][0] =  d * M; A[0][1] = -b * M;
	A[1][0] = -c * M; A[1][1] =  a * M;	
}

template <typename T>
__device__ void Inverse3(T A[3][3]) {
	T a = A[0][0]; T b = A[0][1]; T c = A[0][2];
	T d = A[1][0]; T e = A[1][1]; T f = A[1][2];
	T g = A[2][0]; T h = A[2][1]; T i = A[2][2];
	
	T M = 1 / (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g);
	A[0][0] = M*(e*i-f*h); A[0][1] = M*(c*h-b*i); A[0][2] = M*(b*f-c*e);
	A[1][0] = M*(f*g-d*i); A[1][1] = M*(a*i-c*g); A[1][2] = M*(c*d-a*f);
	A[2][0] = M*(d*h-e*g); A[2][1] = M*(b*g-a*h); A[2][2] = M*(a*e-b*d);
}

//template <typename T>
//__device__ void Inverse4(T A[4][4]) {
//// TODO
//}

template <typename T, int SizeX, int SizeY>
__device__ void CopyToStaticArray(T* Src, T* Dest[SizeX][SizeY]) {
	int id = 0;
	for (int i = 0; i < SizeX; i++) {
		for (int j = 0; j < SizeY; j++) {
			Dest[i][j] = Src[id];
			id++;
		}
	}
}
