
template <typename T, int K1, int K2, int M1, int M2>
class Rational_BSpline_Surface 
{
public:
	__device__ Rational_BSpline_Surface(T* Ptr) {
	}
private:
	static const int N1 = 1 + K1 - M1;
	static const int N2 = 1 + K2 - M2;
	static const int A = N1 + 2 * M1;
	static const int B = N2 + 2 * M2;
	static const int C = (1 + K1) * (1 + K2);
	T Sknot[2 * M1 + N1 + 1]; // knot vectors
	T Tknot[2 * M2 + N2 + 1];
	T W[K1 + 1][K2 + 1];
	T X[K1 + 1][K2 + 1];
	T Y[K1 + 1][K2 + 1];
	T Z[K1 + 1][K2 + 1];
	T U0;
	T U1;
	T V0;
	T V1;
	
};
