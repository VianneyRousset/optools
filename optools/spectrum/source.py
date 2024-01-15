from paramobject import ParametrizedObject, Parameter, parameter
from abc import abstractmethod


class LightSource(ParametrizedObject):
    central_wavelength = Parameter(1550e-9)
    spectral_width = Parameter(100e-9)
    division = Parameter(2001)



    def get_wavelength(self, wavelength=None):

        if isinstance(wavelength, tuple):
            return 


    @abstractmethod
    def get_spectrum(self, wavelength=None):



class UniformLightSource(LightSource):
    pmax = Parameter(1.0)
