using System;

// Reference: The Initial Graphics Exchange Specification (IGES) Version 6.0

namespace FEA.Mesher.IGES
{
    public abstract class IGESEntity{
        public int ParameterId;
        public IGESEntityTypes TypeId;
    }

    public enum IGESEntityTypes
	{
		Circular_Arc = 100,
		Composite_Curve = 102,
		Conic_Arc = 104,
		Copious_Data = 106,
		Two_D_Linear_Path = 11,
		Three_D_Linear_Path = 12,
		Simple_Closed_Planar_Curve = 63,
		Plane = 108,
		Line = 110,
		Parametric_Spline_Curve = 112,
		Parametric_Spline_Surface = 114,
		Point = 116,
		Ruled_Surface = 118,
		Surface_of_Revolution = 120,
		Tabulated_Cylinder = 122,
		Transformation_Matrix = 124,
		Flash = 125,
		Rational_BSpline_Curve = 126,
		Rational_BSpline_Surface = 128,
		Offset_Curve = 130,
		Offset_Surface = 140,
		Boundary = 141,
		Curve_on_a_Parametric_Surface = 142,
		Bounded_Surface = 143,
		Trimmed_Parametric_Surface = 144,
		ManifoldSolidBRepObject = 186,
		Plane_Surface = 190,
		Right_Circular_Cylindrical_Surface = 192,
		Right_Circular_Conical_Surface = 194,
		Spherical_Surface = 196,
		Toroidal_Surface = 198,
        Color_Definition_Entity = 314,
        Associativity_Instance_Entity = 402,
        Property_Entity = 406,
        Vertex = 502,
		Edge = 504,
		Loop = 508,
		Face = 510,
		Shell = 514,
	}

    public enum Rational_BSpline_Surface_Type {
        Unspecified = 0,
        Plane = 1,
        Right_Circular_Cylinder = 2,
        Cone = 3,
        Sphere = 4,
        Torus = 5,
        Surface_of_Revolution = 6,
        Tabulated_Cylinder = 7,
        Ruled_Surface = 8,
        General_Quadric_Surface = 9
    }

    public enum Rational_BSpline_Curve_Type {
        Unspecified = 0,
        Line = 1,
        Circular_Arc = 2,
        Elliptical_Arc = 3,
        Parabolic_Arc = 4,
        Hyperbolic_Arc = 5
    }

}

