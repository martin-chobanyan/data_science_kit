class CUSUM(object):
    """CUSUM algorithm for change detection in time series data

    The CUmulativeSUM algorithm monitors a temporal process and segments the series when
    the target parameter in the probability distribution (e.g. the mean) takes a new value.

    If a singleton value of the series y_i is not beyond the threshold, then its contribution
    to the cumulative deviation sum is zero. If it does go beyond the threshold, then the amount
    of deviation is added to the cumulative sum. If the cumulative sum is above a defined threshold
    then a change point is marked for the target parameter.

    Parameters
    ----------
    mu: float
        The initial "true" value of the target parameter
    margin: float
        The margin for deviation above and below mu
        e.g. if the data generating distribution is Gaussian, this should be ~ 2*standard_dev.
    threshold: float
        The threshold for the cumulative deviation sum. Once it is exceeded, a change point is marked.
    """
    def __init__(self, mu, margin, threshold):
        self.mu = mu
        self.margin = margin
        self.threshold = threshold

    def lower_deviation(self, cum_sum, cur_val):
        """This method updates the cumulative sum for the lower deviation bound"""
        lower_threshold = self.mu - self.margin
        cur_deviation = lower_threshold - cur_val
        return max(0, cum_sum + cur_deviation)

    def upper_deviation(self, cum_sum, cur_val):
        """This method updates the cumulative sum for the upper deviation bound"""
        upper_threshold = self.mu + self.margin
        cur_deviation = cur_val - upper_threshold
        return max(0, cum_sum + cur_deviation)

    def __single_cusum(self, values, upper):
        """This method performs CUSUM for either the upper or lower bound

        If the deviation of mu only matters in one direction, then the cumulative deviation sum
        need only be kept for that direction e.g. only mark the change points for the mean of a
        distribution when it takes a higher value.

        Parameters
        ----------
        values: array_like
            The sequence of values for the series
        upper: bool
            If True, then only the upper deviations are monitored.
            If False, then only the lower deviations are monitored.

        Returns
        -------
        list[int]
            A list of indices in `values` where the target parameter changes.
        """
        cum_sum = 0
        change_points = []
        for i, val in enumerate(values):
            if upper:
                cum_sum = self.upper_deviation(cum_sum, val)
            else:
                cum_sum = self.lower_deviation(cum_sum, val)
            print(i, cum_sum)
            if cum_sum > self.threshold:
                change_points.append(i)
                self.mu = (self.mu + self.margin) if upper else (self.mu - self.margin)
                cum_sum = 0
                print(f'beyond threshold... new mu = {self.mu}')
        return change_points

    def __double_cusum(self, values):
        """This method simultaneously performs CUSUM for both the upper and lower bound

        If the deviation of mu matters in both direction, then two separate cumulative deviation
        sums must be kept. When one of the cumulative sums exceeds the threshold, then a change
        point is marked and mu takes on a new value (depending on which of the sums activated the threshold)

        Parameters
        ----------
        values: array_like
            The sequence of values for the series

        Returns
        -------
        list[int]
            A list of indices in `values` where the target parameter changes.
        """
        lower_cum_sum = 0
        upper_cum_sum = 0
        change_points = []
        for i, val in enumerate(values):
            upper_cum_sum = self.upper_deviation(upper_cum_sum, val)
            lower_cum_sum = self.lower_deviation(lower_cum_sum, val)

            if upper_cum_sum > self.threshold:
                change_points.append(i)
                self.mu += self.margin
                upper_cum_sum = 0

            elif lower_cum_sum > self.threshold:
                change_points.append(i)
                self.mu -= self.margin
                lower_cum_sum = 0
        return change_points

    def __call__(self, values, mode='both'):
        """Perform the CUSUM algorithm on the series of values

        Parameters
        ----------
        values: array_like
            The sequence of values in the series
        mode: str
            This defines which version of the algorithm to apply:
            'upper', 'lower', or 'both' (default='both')

        Returns
        -------
        list[int]
            A list of indices where the target parameter changes value.

        For further details see the documentation of __single_cusum and __double_cusum.
        """
        if mode == 'upper':
            return self.__single_cusum(values, True)
        elif mode == 'lower':
            return self.__single_cusum(values, False)
        elif mode == 'both':
            return self.__double_cusum(values)
        else:
            raise ValueError(f'Argument "mode" can only have values: "both", "upper", and "lower".')
