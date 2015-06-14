
__device__ int Factorial(int Value) {
	int Ans = 1;
	for (int i = Value, i > 0, i--) {
		Ans = Ans * i;
	}
	return Ans;
}

template <typename T>
__device__ T VolumeIntegrate(int Alpha, int Beta, int Gamma, int Delta, T Volume) { // volume elements
	int Numerator = Factorial(Alpha) * Factorial(Beta) * Factorial(Gamma) * Factorial(Delta);
	int Denominator = Factorial(Alpha + Beta + Gamma + Delta + 3);
	T ans = (T) (6 * Numerator / Denomator) * Volume;
	return ans;
}

template <typename T>
__device__ T AreaIntegrate(int Alpha, int Beta, int Gamma, T Area) { // area elements
	int Numerator = Factorial(Alpha) * Factorial(Beta) * Factorial(Gamma);
	int Denominator = Factorial(Alpha + Beta + Gamma + Delta + 2);
	T ans = (T) (2 * Numerator / Denomator) * Area;
	return ans;
}

template <typename T>
__device__ T DistanceIntegrate(int Alpha, int Beta, T Length) { // line elements
	int Numerator = Factorial(Alpha) * Factorial(Beta);
	int Denominator = Factorial(Alpha + Beta + 1);
	T ans = (T) (Numerator / Denomator) * Length;
	return ans;
}
