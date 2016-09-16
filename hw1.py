import numpy as np
from random import choice

class Learner():
	def __init__(self, algorithm, label_gen, experts):
		if algorithm == "WMA":
			self.algorithm = 0
		elif algorithm == "RWMA":
			self.algorithm = 1
		else:
			print "check algorithm spelling"

		if label_gen == "stochastic":
			self.labels = 0
		elif label_gen == "deterministic":
			self.labels = 1
		elif label_gen == "adversarial":
			self.labels = 2
		else:
			print "check label generator spelling"

		self.experts = experts

	def expertAdvice(self,i):
		h = [0]*self.experts
		h[0] = 1
		h[1] = -1
		if i%2 == 0:	#odd
			h[2] = -1
		else: 			#even
			h[2] = 1
		return h

	def predictor(self, h,w):
		if self.algorithm == 0:
			dotp = np.dot(h,w)
			if dotp < 0:
				y_hat = -1
			else:
				y_hat = 1
		else:
			print "TO BE WRITTEN"
		return y_hat

	def true_label(self, h, w):
		if self.labels ==0:
			y  = choice([-1,1])
		elif self.labels ==1:
			#always winning 
			y  = 1
		else:
			# adversary
			y = np.dot(h,w)
			if y > 0:
				y = -1
			else:
				y = 1
		return y

	def learn(self,T, eta):
		weights = [1]*self.experts
		if self.algorithm ==0:
			for i in range(T):
				hypothesis = self.expertAdvice(i)
				y_hat = self.predictor(hypothesis,weights)
				y = self.true_label(hypothesis,weights)
#				print "yh", y_hat
#				print "yo", y
				loss = self.calculateLoss(y,y_hat)
				weights = self.updateWeights(weights,loss,hypothesis,eta)

	def calculateLoss(self, y,y_hat):
		l = y_hat - y
		if l ==0:
			l = 0
		else:
			l = 1
#		print l
		return l

	def updateWeights(self,w,loss,hypothesis,eta):
		return w



def main():
	T = 10
	eta = 0.5
	experts = 3
	"""
	WMA, RWMA
	stochastic,deterministic,adversarial
	"""
	WMA = Learner("WMA","stochastic", experts)
	WMA.learn(T,eta)

if __name__ == "__main__":
	main()