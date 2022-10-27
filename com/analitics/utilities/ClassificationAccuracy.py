def precision(true_positive, false_positive, decimals=3):                    # Positive Predictive Value
	"""
	Also known as Positive Predictive Value.
	:param true_positive:
	:param false_positive:
	:param decimals:
	:return:
	"""
	den = len(true_positive) + len(false_positive)
	return round(len(true_positive) / den, decimals) if den > 0 else 0.0


def negative_predictive_value(true_negative, false_negative, decimals=3):     # Negative Predicted Value
	"""
	
	:param true_negative:
	:param false_negative:
	:param decimals:
	:return:
	"""
	den = len(true_negative) + len(false_negative)
	return round(len(true_negative) / den, decimals) if den > 0 else 0.0


def recall(true_positive, false_negative, decimals=3):                       # Sensitivity, True Positive Rate
	"""
	
	:param true_positive:
	:param false_negative:
	:param decimals:
	:return:
	"""
	den = len(true_positive) + len(false_negative)
	return round(len(true_positive) / den, decimals) if den > 0 else 0.0


def fall_out(false_positive, true_negative, decimals=3):                     # False Positive Rate
	"""
	
	:param false_positive:
	:param true_negative:
	:param decimals:
	:return:
	"""
	den = len(false_positive) + len(true_negative)
	return round(len(false_positive) / den, decimals) if den > 0 else 0.0


def specificity(true_negative, false_positive, decimals=3):                  # True Negative Rate
	"""
	
	:param true_negative:
	:param false_positive:
	:param decimals:
	:return:
	"""
	den = len(true_negative) + len(false_positive)
	return round(len(true_negative) / den, decimals) if den > 0 else 0.0


def miss_rate(false_negative, true_positive, decimals=3):                 # False Negative Rate
	"""
	
	:param false_negative:
	:param true_positive:
	:param decimals:
	:return:
	"""
	den = len(false_negative) + len(true_positive)
	return round(len(false_negative) / den, decimals) if den > 0 else 0.0


def f1(precision_val, recall_val, decimals=3):
	"""
	
	:param precision_val:
	:param recall_val:
	:param decimals:
	:return:
	"""
	den = precision_val + recall_val
	return round((2 * precision_val * recall_val) / den, decimals) if den > 0 else 0.0


