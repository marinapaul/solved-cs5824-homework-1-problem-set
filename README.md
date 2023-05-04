Download Link: https://assignmentchef.com/product/solved-cs5824-homework-1-problem-set
<br>
<h1></h1>

<ol>

 <li> Show what the recursive decision tree learning algorithm would choose for the first split of the following dataset:</li>

</ol>

Assume that the criterion for deciding the best split is entropy reduction (i.e., information gain). If there are any ties, choose the first feature to split on tied for the best score. Show your calculations in your response.

(Hint: this dataset is one of the test cases in the programming assignment, so you should be able to use your answer here to debug your code.)

<ol start="2">

 <li>A Bernoulli distribution has the following likelihood function for a data set D:</li>

</ol>

<em>p</em>(D|<em>θ</em>) = <em>θ<sup>N</sup></em><sup>1</sup>(1 − <em>θ</em>)<em><sup>N</sup></em><sup>0</sup><em>,                                                                           </em>(1)

where <em>N</em><sub>1 </sub>is the number of instances in data set D that have value 1 and <em>N</em><sub>0 </sub>is the number in D that have value 0. The maximum likelihood estimate is

<em>.                                                                                 </em>(2)

<ul>

 <li> Derive the maximum likelihood estimate above by solving for the maximum of the likelihood. I.e., show the mathematics that get from Equation (1) to Equation (2).</li>

 <li> Suppose we now want to maximize a posterior likelihood</li>

</ul>

<em>,                                                                     </em>(3)

where we use the Bernoulli likelihood and a (slight variant<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> of a) symmetric Beta prior over the Bernoulli parameter

<em>p</em>(<em>θ</em>) ∝ <em>θ<sup>α</sup></em>(1 − <em>θ</em>)<em><sup>α</sup>.                                                                         </em>(4)

Derive the maximum posterior mean estimate.

<h1>Programming Assignment</h1>

For this homework, you will build two text categorization classifiers: one using naive Bayes and the other using decision trees. You will write general code for cross-validation that will apply to either of your classifiers.

<strong>Data and starter code: </strong>In the HW1 archive, you should find the 20newsgroups data set (also available from the original source <a href="http://qwone.com/~jason/20Newsgroups/">http://qwone.com/</a><a href="http://qwone.com/~jason/20Newsgroups/">~</a><a href="http://qwone.com/~jason/20Newsgroups/">jason/20Newsgroups/</a><a href="http://qwone.com/~jason/20Newsgroups/">)</a>. This data set, whose origin is some-

what fuzzy, consists of newsgroup posts from an earlier era of the Internet. The posts are in different categories, and this data set has become a standard benchmark for text classification methods.

The data is represented in a bag-of-words format, where each post is represented by what words are present in it, without any consideration of the order of the words.

We have also provided a unit test class in tests.py, which contains unit tests for each type of learning model. These unit tests may be easier to use for debugging in an IDE like PyCharm than the iPython notebook. A successful implementation should pass all unit tests and run through the entire iPython notebook

without issues. You can run the unit tests from a *nix command line with the command python -m unittest -v tests

or you can use an IDE’s unit test interface. These tests are not foolproof, so it’s possible for code that does not meet the requirements for full credit to pass the tests (and, though it would be surprising, it may be possible for full credit code to fail the tests).

Before starting all the tasks, examine the entire codebase. Follow the code from the iPython notebook to see which methods it calls. Make sure you understand what all of the code does. Your required tasks follow.

<ol>

 <li> Examine the iPython notebook test predictors.ipynb. This notebook uses the learning algorithms and predictors you will implement in the first part of the assignment. Read through the data-loading code and the experiment code to make sure you understand how each piece works.</li>

 <li> Examine the function calculate information gain in decision tree.py. The function takes in training data and training labels and computes the information gain for each feature. That is, for each feature dimension, compute</li>

</ol>

<em>G</em>(<em>Y,X<sub>j</sub></em>) = <em>H</em>(<em>Y </em>) − <em>H</em>(<em>Y </em>|<em>X<sub>j</sub></em>)

= −<sup>X</sup>Pr(<em>Y </em>= <em>y</em>)logPr(<em>Y </em>= <em>y</em>)


<em>y                                                                                                                                                                                                            </em>(5)

<h2>                                                                X                              X</h2>

Pr(<em>X<sub>j </sub></em>= <em>x<sub>j</sub></em>)                      Pr(<em>Y </em>= <em>y</em>|<em>X<sub>j </sub></em>= <em>x<sub>j</sub></em>)logPr(<em>Y </em>= <em>y</em>|<em>X<sub>j </sub></em>= <em>x<sub>j</sub></em>)<em>.</em>

<em>x<sub>j                                                                                           </sub>y</em>

Your function should return the vector

[<em>G</em>(<em>Y,X</em><sub>1</sub>)<em>,…,G</em>(<em>Y,X<sub>d</sub></em>)]<sup>&gt;</sup><em>.                                                                           </em>(6)

<strong>Algorithm 1 </strong>Recursive procedure to grow a classification tree

<table width="0">

 <tbody>

  <tr>

   <td colspan="3" width="624">1: <strong>function </strong>FITTREE(D, depth)2:                  <strong>if </strong>not worth splitting (because D is all one class or max depth is reached) <strong>then</strong>3:          node.prediction ← argmax<em><sub>c </sub></em><sup>P</sup><sub>(<em>x</em></sub><em><sub>,y</sub></em><sub>)∈D </sub><em>I</em>(<em>y </em>= <em>c</em>) 4: <strong>return </strong>node</td>

  </tr>

  <tr>

   <td width="43">5:</td>

   <td width="375"><em>w </em>← argmax<em><sub>w</sub></em>0 <em>G</em>(<em>Y,X<sub>w</sub></em>)</td>

   <td width="207"><em>. </em>See Equation (5</td>

  </tr>

  <tr>

   <td width="43">6:</td>

   <td width="375">node.test ← <em>w</em></td>

   <td width="207"> </td>

  </tr>

  <tr>

   <td width="43">7:</td>

   <td width="375">node.left ← FITTREE(D<em><sub>L</sub></em>, depth+1)</td>

   <td width="207"><em>. </em>where D<em><sub>L </sub></em>:= {(<em>x</em><em>,y</em>) ∈ D|<em>x<sub>w </sub></em>= 0}</td>

  </tr>

  <tr>

   <td width="43">8:</td>

   <td width="375">node.right ← FITTREE(D<em><sub>R</sub></em>, depth+1)</td>

   <td width="207"><em>. </em>where D<em><sub>R </sub></em>:= {(<em>x</em><em>,y</em>) ∈ D|<em>x<sub>w </sub></em>= 1}</td>

  </tr>

  <tr>

   <td width="43">9:</td>

   <td width="375"><strong>return </strong>node</td>

   <td width="207"> </td>

  </tr>

 </tbody>

</table>

)

You will use this function to do feature selection and as a subroutine for decision tree learning. Note how the function avoids loops over the dataset and only loops over the number of classes. Follow this style to avoid slow Python loops; use numpy array operations whenever possible.

<ol start="3">

 <li>(5 points) Finish the functions naive bayes train and naive bayes predict in naive bayes.py. The training algorithm should find the maximum likelihood parameters for the probability distribution</li>

</ol>

Pr(<em>y<sub>i </sub></em>= <em>c</em>)<sup>Q</sup><em><sub>w</sub></em>∈<em><sub>W </sub></em>Pr(<em>x<sub>iw</sub></em>|<em>y<sub>i </sub></em>= <em>c</em>)

Pr(<em>y<sub>i </sub></em>= <em>c</em>|<em>x</em><em><sub>i</sub></em>) =     <em>.</em>

Pr(<em>x<sub>i</sub></em>)

Make sure to use log-space representation for these probabilities, since they will become very small, and notice that you can accomplish the goal of naive Bayes learning without explicitly computing the prior probability Pr(<em>x<sub>i</sub></em>). In other words, you can predict the most likely class label without explicitly computing that quantity.

Implement additive smoothing (<a href="https://en.wikipedia.org/wiki/Additive_smoothing">https://en.wikipedia.org/wiki/Additive_smoothing</a><a href="https://en.wikipedia.org/wiki/Additive_smoothing">)</a> for your naive Bayes learner. One natural way to do this is to let the input parameter params simply be the prior count for each word. For a parameter <em>α</em>, this would mean your maximum likelihood estimates for any Bernoulli variable <em>X </em>would be

(# examples where <em>X</em>) + <em>α</em>

Pr(<em>X</em>) =    <em>.</em>

(Total # of examples) + 2<em>α</em>

Notice that if <em>α </em>= 0, you get the standard maximum likelihood estimate. Also, make sure to multiply <em>α </em>by the total number of possible outcomes in the distribution. For the label variables in the 20newsgroups data, there are 20 possible outcomes, and for the word-presence features, there are two.

<ol start="4">

 <li>(5 points) Finish the functions recursive tree train and decision tree predict in decision tree.py. Note that recursive tree train is a helper function used by decision tree train, which is already completed for you. You’ll have to design a way to represent the decision tree in the model object. Your training algorithm should take a parameter that is the maximum depth of the decision tree, and the learning algorithm should then greedily grow a tree of that depth. Use the information-gain measure to determine the branches (hint: you’re welcome to use the calculate information gain function). Algorithm 1 is abstract pseudocode describing one way to implement decision tree training. You are welcome to deviate from this somewhat; there are many ways to correctly implement such procedures.</li>

</ol>

The pseudocode suggests building a tree data structure that stores in each node either (1) a prediction or (2) a word to split on and child nodes. The pseudocode also includes the formula for the entropy criterion for selecting which word to split on.

The prediction function should have an analogous recursion, where it receives a data example and a node. If the node has children, the function should determine which child to recursively predict with. If it has no children, it should return the prediction stored at the node.

<ol start="5">

 <li>(5 points) Finish the function cross validate in crossval.py, which takes a training algorithm, a prediction algorithm, a data set, labels, parameters, and the number of folds as input and performs cross-fold validation using that many folds. For example, calling</li>

</ol>

params[’alpha’] = 1.0 score = cross_validate(naive_bayes_train, naive_bayes_predict, train_data, train_labels, 10, params)

will compute the 10-fold cross-validation accuracy of naive Bayes using regularization parameter <em>α </em>= 1<em>.</em>0.

The cross-validation should split the input data set into folds subsets. Then iteratively hold out each subset: train a model using all data <em>except </em>the subset and evaluate the accuracy on the held-out subset. The function should return the average accuracy over all folds splits.

Some code to manage the indexing of the splits is included. You are welcome to change it if you prefer a different way of organizing the indexing.

Once you complete this last step, you should be able to run the notebook cv predictors.ipynb, which should use cross validation to compare decision trees to naive Bayes on the 20-newsgroups task. Naive Bayes should do be much more accurate than decision trees, but the cross-validation should find a decision tree depth that performs a bit better than the depth hard coded into test predictors.ipynb.

<a href="#_ftnref1" name="_ftn1">[1]</a> For convenience, we are using the exponent of <em>α </em>instead of the standard <em>α</em>−1.