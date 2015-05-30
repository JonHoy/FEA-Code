
// class for 3D polynomial

template <typename T, int SizeX, int SizeY = 1, int SizeZ = 1>
class Polynomial {
public:
	Polynomial() {
	
	}
	T Evaluate(T ValX, T ValY = 1, T ValZ = 1) {
		T sum = 0;
		for (int i = 0; i < SizeX; i++) {
			for (int j = 0; j < SizeY; j++) {
				for (int k = 0; k < SizeZ; k++)				
					T a = (*this)(i,j,k);
					sum += a*Pow(ValX,i)*Pow(ValY,j)*Pow(ValZ,k);	
			}
		}
		return sum;
	}
	T Differentiate(const int Dim, T ValX, T ValY = 1, T ValZ = 1) {
	
	}
	T& operator()(int i) {
		return Coeffs[i];
	}
	T& operator()(int i, int j) {
		return Coeffs[i * SizeY + j];
	}
	T& operator()(int i, int j, int k) {
		return Coeffs[i * SizeZ * SizeY + j * SizeZ + k];
	}
private:
	T Coeffs[SizeX * SizeY * SizeZ];
	T Pow(T a, int b) {
		T ans = 1;
		for (int i = 0; i < b; i++)
			ans *= a
		return ans;		
	}
};

