#include "lbfgs.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

class rosenbrock : public cpu_cost_function
{
public:
	rosenbrock()
	: cpu_cost_function(2) {}

	void cpu_f(const floatdouble *h_x, floatdouble *h_y)
	{
		const floatdouble x0 = h_x[0];
		const floatdouble x1 = h_x[1];

		// f = (1-x0)^2 + 100 (x1-x0^2)^2

		const floatdouble a = (1.0f - x0);
		const floatdouble b = (x1 - x0 * x0) ;

		*h_y = (a*a) + 100.0f * (b*b);
	}

	void cpu_gradf(const floatdouble *h_x, floatdouble *h_grad)
	{
		const floatdouble x0 = h_x[0];
		const floatdouble x1 = h_x[1];

		// df/dx0 = -2 (1-x0) - 400 (x1-x0^2) x0
		// df/dx1 = 200 (x1 - x0^2)

		h_grad[0] = -2.0f * (1.0f - x0) - 400.0f * x0 * (x1 - x0*x0);
		h_grad[1] = 200.0f * (x1 - x0*x0);
	}

	void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf)
	{
		cpu_f(h_x, h_f);
		cpu_gradf(h_x, h_gradf);
	}
};

bool test(floatdouble x0, floatdouble x1, floatdouble epsilon)
{
	float xstart[] = { float(x0), float(x1) };

	rosenbrock rcf;
	lbfgs minimizer(rcf);
	minimizer.setGradientEpsilon(1e-3f);

	lbfgs::status stat = minimizer.minimize_with_host_x(xstart);

	floatdouble e0 = std::abs(xstart[0] - 1.0f);
	floatdouble e1 = std::abs(xstart[1] - 1.0f);

	if (e0 > epsilon || e1 > epsilon)
	{
		cerr << "Ended because: " << minimizer.statusToString(stat).c_str() << endl;
		cerr << "Starting point (" << x0 << ", " << x1 << ")" << endl;
		// cerr << "x = " << xstart[0] << ", err(x) = " << e0 << endl;
		// cerr << "y = " << xstart[1] << ", err(y) = " << e1 << endl;
		// cerr << "Max. allowed error: " << epsilon << endl;
		return false;
	}

	return true;
}

int main (int argc, char const *argv[])
{
	for (int i = -4; i < 5; ++i)
	{
		for (int j = -4; j < 5; ++j)
		{
			if (!test(floatdouble(i), floatdouble(j), 1e-2f))
				exit(EXIT_FAILURE);
		}
	}

	cout << "Tests successful." << endl;

	return EXIT_SUCCESS;
}
