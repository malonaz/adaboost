# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), and 6.034 staff

from math import log as ln
from utils import *


#### BOOSTING (ADABOOST) #######################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    n = len(training_points)
    weights = {}
    for training_point in training_points:
        weights[training_point] = make_fraction(1,n)
    return weights

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    n = len(point_to_weight)
    classifier_to_error_rates = {}
    for classifier in classifier_to_misclassified:
        weighted_misclassified_points = map(lambda point: point_to_weight[point],\
                                            classifier_to_misclassified[classifier])
        classifier_to_error_rates[classifier] = make_fraction(sum(weighted_misclassified_points))
    return classifier_to_error_rates

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    list_classifier_errors =  list(classifier_to_error_rate.items())
    list_classifier_errors =sorted(list_classifier_errors,\
                                   key = lambda (c,e): c) # get alphabetical order
    if use_smallest_error:
        list_classifier_errors = sorted(list_classifier_errors,\
                                        key = lambda (c,e): e) # now sort on errors
        if list_classifier_errors[0][1] < make_fraction(1,2):
            return list_classifier_errors[0][0]
        raise NoGoodClassifiersError
    distance_from_half = map(lambda (c,e): (c,abs(make_fraction(1,2)-e)), list_classifier_errors)
    distance_from_half = sorted(distance_from_half,\
                                key =lambda (c,e): e, reverse = True) #sort on errors but reversed
    if distance_from_half[0][1] != make_fraction(0,1):
        return distance_from_half[0][0]
    raise NoGoodClassifiersError
    
    

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 0:
        return INF
    if error_rate ==1:
        return -INF
    return .5*ln((1-error_rate)/error_rate)

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""

    points_score = dict(map(lambda t_p: (t_p,0),training_points))
    total_voting_power = 0
    for (classifier,voting_power) in H:
        #total_voting_power += voting_power
        mis_classified = set(classifier_to_misclassified[classifier])
        cor_classified = set(training_points) - mis_classified
        for mis_point in mis_classified:
            points_score[mis_point] += voting_power
        for cor_point in cor_classified:
            points_score[cor_point] -= voting_power
    list_mis_points = []
    for (point,count) in points_score.items():
        if count >= 0:                                           
            list_mis_points.append(point)
    return set(list_mis_points)
        

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    list_mis_points = get_overall_misclassifications(H,training_points,classifier_to_misclassified)
    return len(list_mis_points) <= mistake_tolerance

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for point in point_to_weight:
        old_weight = make_fraction(point_to_weight[point])
        if point in misclassified_points: #point is misclassified
            new_weight = make_fraction(.5)*make_fraction(1/error_rate)*old_weight
        else:
            new_weight = make_fraction(.5)*make_fraction(1/(1-error_rate))*old_weight
        point_to_weight[point] = new_weight
    return point_to_weight

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    
    point_to_weight = initialize_weights(training_points)
    H = []
    while max_rounds > 0:
        classifier_to_error_rate = calculate_error_rates(point_to_weight,\
                                                           classifier_to_misclassified)
        try:
            best_classifier = pick_best_classifier(classifier_to_error_rate,\
                                               use_smallest_error)# try statement?
        except:
            return H
        error_rate = classifier_to_error_rate[best_classifier]
        alpha = calculate_voting_power(error_rate)
        H.append((best_classifier,alpha))
        misclassified_points = get_overall_misclassifications(H, training_points,\
                                                    classifier_to_misclassified)
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H
        mis_points_curr_class = classifier_to_misclassified[best_classifier]
        point_to_weight = update_weights(point_to_weight, mis_points_curr_class,\
                                         classifier_to_error_rate[best_classifier]) #update weights
        max_rounds -= 1
    return H
        
#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
