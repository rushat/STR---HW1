import numpy as np
from random import choice
from random import random
import matplotlib.pyplot as plt
class Learner():
	def __init__(self, algorithm, label_gen, experts):
		if algorithm == "WMA":
			self.algorithm = 0
			self.algorithm_name = algorithm
		elif algorithm == "RWMA":
			self.algorithm = 1
			self.algorithm_name = algorithm
		else:
			print "check algorithm spelling"

		if label_gen == "realWorld":
			self.labels = 0
			self.label_name = label_gen
		else:
			print "check label generator spelling"

		self.experts = experts
		self.weights = [1]*experts
		self.loss = []
		self.total_loss =[]
		self.expert_loss = [[] for _ in range(experts)]
		self.cumulative_loss = [[] for _ in range(experts)]
		self.avg_regret = []
		self.weather = ["sunny","notSunny"]
		self.prob_sunny = 0.7
		self.prob_winning_sunny = 0.7
		self.prob_winning_not_sunny = 0.2

	def expertAdvice(self,t,weather,sunny_counter,not_sunny_counter,win_counter,sunny_win_counter,not_sunny_win_counter):
		h = [0]*self.experts
		h[0] = 1
		h[1] = -1
		if t%2 == 0:	#odd
			h[2] = -1
		else: 			#even
			h[2] = 1
		
		if weather == "sunny":
			if sunny_counter ==0:
				h[3] = 1
			else:
				p_win_sunny = (sunny_win_counter*1.)/sunny_counter #from bayes rule
				print "p_win_sunny",p_win_sunny
				if p_win_sunny>=0.5:
					h[3] = 1
				else:
					h[3] = -1

		else:
			if not_sunny_counter ==0:
				h[3] = 1
			else:
				p_win_not_sunny = (not_sunny_win_counter*1.)/not_sunny_counter
				print "p_win_not_sunny",p_win_not_sunny
				if p_win_not_sunny>=0.5:
					h[3] = 1
				else:
					h[3] = -1
		return h

	def predictor(self, h, ind):
		if self.algorithm == 0:
			dotp = np.dot(h,self.weights)
		else:
			dotp = np.dot(h[ind],self.weights[ind])
		if dotp < 0:
			y_hat = -1
		else:
			y_hat = 1
		return y_hat

	def true_label(self, weather):
		prob_winning = random()
		
		if weather=="sunny":
			if prob_winning<self.prob_winning_sunny:
				y = 1
			else:
				y = -1
		else:
			if prob_winning<self.prob_winning_not_sunny:
				y = 1
			else:
				y = -1
		return y

	def getWeather(self):
		prob_weather = random()
		if prob_weather<self.prob_sunny:
			weather = "sunny"
		else:
			weather = "notSunny"
		return weather

	def learn(self,T, eta):
		self.weights = [1]*self.experts
		phi = 0
		ind = 0
		sunny_counter = 0
		not_sunny_counter =0
		win_counter = 0
		sunny_win_counter = 0
		not_sunny_win_counter = 0
		cumulative_loss = [0]*self.experts
		for i in range(T):
			#get weather
			weather = self.getWeather()

			hypothesis = self.expertAdvice(i,weather,sunny_counter, not_sunny_counter,win_counter,sunny_win_counter,not_sunny_win_counter)
			
			if self.algorithm == 1:
				phi = np.sum(self.weights)
				ind = np.argmax(np.random.multinomial(1, np.divide(self.weights,phi)))
			
			y_hat = self.predictor(hypothesis,ind)
			y = self.true_label(weather)
			
			#counter update 
			if y ==1:
				win_counter +=1
			if weather == "sunny":
				sunny_counter +=1
				if y==1:
					sunny_win_counter+=1
			else:
				not_sunny_counter+=1
				if y==1:
					not_sunny_win_counter+=1
			# print "win_counter",win_counter
			# print "sunny_counter",sunny_counter
			# print "sunny_win_counter",sunny_win_counter
			# print "not_sunny_counter",not_sunny_counter
			# print "not_sunny_win_counter",not_sunny_win_counter
			if sunny_counter!=0 and not_sunny_counter!=0:
				p_win_not_sunny = (not_sunny_win_counter*1.)/not_sunny_counter
				p_win_sunny = (sunny_win_counter*1.)/sunny_counter
				print "p_win_sunny",p_win_sunny
				print "p_win_not_sunny",p_win_not_sunny 

			loss = self.calculateLoss(y,y_hat)
			self.storeAlgorithmLoss(loss)
			for ind in range(self.experts):
				self.storeExpertLoss(y,ind,hypothesis[ind])
				cumulative_loss[ind] = self.storeCumulativeLoss(ind)
			self.calculateAvgRegret(cumulative_loss,i)
			self.weights = self.updateWeights(hypothesis,y,eta)
			"""print "y_hat", y_hat
			print "expert 3 prediction",self.expertAdvice(i)[2]
			print "y_actual", y
			print "loss", loss
			print "expert3 loss", self.expert_loss[2][-1]
			print "weights", self.weights"""
		#print "regrets", self.regret
		print "Total loss", np.sum(self.loss)

	def storeAlgorithmLoss(self,loss):
		self.loss.append(loss)
		self.total_loss.append(np.sum(self.loss))

	def storeExpertLoss(self,y,ind,hypothesis):
		loss_exp = self.calculateLoss(y,hypothesis)
		self.expert_loss[ind].append(loss_exp)

	def storeCumulativeLoss(self,ind):
		cumulative_loss = [0]*self.experts
		cumulative_loss[ind] = np.sum(self.expert_loss[ind])
		self.cumulative_loss[ind].append(cumulative_loss[ind])
		return cumulative_loss[ind]

	def calculateAvgRegret(self,cumulative_loss,i):
		self.avg_regret.append((np.sum(self.loss) - min(cumulative_loss))/(i+1.))

	def calculateLoss(self, y,y_hat):
		l = y_hat-y
		if l ==0:
			l = 0
		else:
			l = 1
		return l

	def updateWeights(self,h,y,eta):
		for i in range(self.experts):
			loss = self.calculateLoss(y,h[i])
			if loss != 0:
				self.weights[i] = self.weights[i]*(1-eta)
		return self.weights

	def plotLossGraph(self,T):
		time = np.arange(T)
		#f, axarr = plt.subplots(3, sharex=True)
		plt.title('Loss Plot for {} and {} case'.format(self.algorithm_name, self.label_name))
		plt.plot(time,self.total_loss, label='Algorithm loss')	
		for i in range(self.experts):
			plt.plot(time,self.cumulative_loss[i], label='Expert {}'.format(i+1))
		plt.legend()
		plt.ylabel('loss')
		plt.xlabel('time')
		
	def plotRegretGraph(self,T):
		time = np.arange(T)
		plt.figure()
		plt.title('Average Regret Plot for {} and {} case'.format(self.algorithm_name, self.label_name))
		plt.plot(time,self.avg_regret)
		plt.ylabel('avg regret')
		plt.xlabel('time')
		plt.show()

def main():
	T = 100
	eta = 0.2

	allplots = False
	experts = 4
	param1 = ["WMA","RWMA"]
	param2 = ["realWorld"]

	if allplots:
		for i in param1:
			for j in param2:
				predict = Learner(i,j, experts)
				predict.learn(T,eta)
				predict.plotLossGraph(T)
				predict.plotRegretGraph(T)
	else:
		predict = Learner(param1[0],param2[0], experts)
		predict.learn(T,eta)
		predict.plotLossGraph(T)
		predict.plotRegretGraph(T)

if __name__ == "__main__":
	main()