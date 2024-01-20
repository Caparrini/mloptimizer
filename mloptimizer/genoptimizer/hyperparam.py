class Hyperparam(object):
    """
    Class to define a hyperparam to optimize. It defines the name, min value, max value and type.
    This is used to control the precision of the hyperparam and avoid multiple evaluations
    with close values of the hyperparam due to decimal positions.


    Attributes
    ----------
    name : str
        Name of the hyperparam. It will be used as key in a dictionary
    min_value : int
        Minimum value of the hyperparam
    max_value : int
        Maximum value of the hyperparam
    type : type
        Type of the hyperparam (int, float, 'nexp', 'x10')
    denominator : int, optional (default=100)
        Optional param in case the type=float
    values_str : list, optional (default=[])
        List of string with possible values (TODO)
    """

    def __init__(self, name: str, min_value: int, max_value: int, hyperparam_type,
                 denominator: int = 100, values_str: list = None):
        """
        Creates object Hyperparam.

        Parameters
        ----------
        name : str
            Name of the hyperparam. It will be used as key in a dictionary
        min_value : int
            Minimum value of the hyperparam
        max_value : int
            Maximum value of the hyperparam
        type : type
            Type of the hyperparam (int, float, 'nexp', 'x10')
        denominator : int, optional (default=100)
            Optional param in case the type=float
        values_str : list, optional (default=[])
            List of string with possible values (TODO)
        """
        if values_str is None:
            values_str = []
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.type = hyperparam_type
        self.denominator = denominator
        self.values_str = values_str

    def correct(self, value: int):
        """
        Returns the real value of the hyperparam in case some mutation could surpass the limits.
            1) Verifies the input is int
            2) Enforce min and max value
            3) Apply the type of value

        Parameters
        ----------
        value : int
            Value to correct

        Returns
        -------
        ret : int, float
            Corrected value
        """
        # Input value must be int
        value = int(value)
        ret = None
        # Verify the value is in range
        if value > self.max_value:
            value = self.max_value
        elif value < self.min_value:
            value = self.min_value
        # Apply the type of value
        if self.type == int:
            ret = value
        elif self.type == float:
            ret = float(value) / self.denominator
            # ret = round(value, self.decimals)
        elif self.type == "nexp":
            ret = 10 ** (-value)
        elif self.type == "x10":
            ret = value * 10
        return ret

    def __eq__(self, other_hyperparam):
        """Overrides the default implementation"""
        equals = (self.name == other_hyperparam.name and self.min_value == other_hyperparam.min_value and
                  self.type == other_hyperparam.type and self.denominator == other_hyperparam.denominator and
                  self.max_value == other_hyperparam.max_value)
        return equals

    def __str__(self):
        """Overrides the default implementation"""
        if self.type is str:
            type_str = "'{}'".format(self.type)
        else:
            type_str = self.type.__name__

        if self.type == float:
            hyperparam_str = "Hyperparam('{}', {}, {}, {}, {})".format(
                self.name,
                self.min_value,
                self.max_value,
                type_str,
                self.denominator
            )
        else:
            hyperparam_str = "Hyperparam('{}', {}, {}, {})".format(
                self.name,
                self.min_value,
                self.max_value,
                type_str
            )

        return hyperparam_str

    def __repr__(self):
        """Overrides the default implementation"""
        return self.__str__()

