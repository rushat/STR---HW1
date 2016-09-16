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
				print "yh", y_hat
				print "yo", y
				loss = self.calculateLoss(y,y_hat)
				weights = self.updateWeights(weights,hypothesis,y,eta)
				print "loss", loss
				print "weights", weights

	def calculateLoss(self, y,y_hat):
		l = y_hat-y
		if l ==0:
			l = 0
		else:
			l = 1
		return l

	def updateWeights(self,w,h,y,eta):
		for i in range(self.experts):
			loss = self.calculateLoss(y,h[i])
			if loss != 0:
				w[i] = w[i]*(1-eta)
		return w

	def plotGraph():



def main():
	T = 100
	eta = 0.1
	experts = 3
	"""
	WMA, RWMA
	stochastic,deterministic,adversarial
	"""
	WMA = Learner("WMA","adversarial", experts)
	WMA.learn(T,eta)

if __name__ == "__main__":
	main()