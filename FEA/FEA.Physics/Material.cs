using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FEA.Physics
{
    class Material
    {
         
        Property Density; // rho
        Property Viscosity; // mu
        Property ThermalConductivity; // k
        Property SpecificHeatCapacity; //Cp

    };
    class Property { 
        
    };// represents a physical property such as 
    enum PropertyType {
        Constant, // property values is treated as a constant value independent of other variables
        Equation, // property value is approximated by an analytical expression with input variables
        LookupTable // property value is approximated by a lookup table
    };
}
