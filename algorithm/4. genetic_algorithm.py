# GENETIC ALGORITHM
import random
random.seed(42)

'''
encode chromosome:
	- chromosome is a permutation of [-K+1, -K+2, ..., 0, 1, 2, .., N], length of K + N
	- where [-K+1, -K+2, .., 0] represent truck and [1,2,3,..., N] represent Node
	or gene <= 0 represent truck and gene > 0 represent Node
	- note that chromosome allways start at -K+1 so in the actual code,
	chromosome is a permutation of [-K+2, -K+3, ..., 0, 1, 2, ..., N] length of K + N - 1
decode chromosome:
	- loop through chromosome, when gene <= 0 that mean we go to next truck and all node > 0 after that is the Route of that truck
'''

class Individual:
	def __init__(self, N, K, distance_matrix, chromosome = None):

		self.N = N 
		self.K = K
		#len of chromosome
		self.n = N + K - 1

		#if chromosome = None, create random solution
		if chromosome == None:
			self.chromosome = [-i for i in range(K-1)]
			self.chromosome += [i+1 for i in range(N)]
			random.shuffle(self.chromosome)
			
		else:
			self.chromosome = chromosome


		self.distance_matrix = distance_matrix

		self.fitness = 0


		self.prob = 0

	
	# calc fitness base on Route
	def calc_fitness(self):

		# Routes i represent truck {i+1} Route
		self.Routes = [[] for _ in range(self.K)]

		# start at the first truck
		index = 0
		self.Routes[0].append(0)

		for gene in self.chromosome:
			#if gene <= 0, go to next truck
			if gene <= 0:			
				index += 1
				#truck allways start at index 0
				self.Routes[index].append(0)
			else:
				self.Routes[index].append(gene)


		#fitness i represent cost of truck {i+1}
		fitnesses = [0 for _ in range(self.K)]
		for i, route in enumerate(self.Routes):
			for j in range(1, len(route)):
				fitnesses[i] += self.distance_matrix[route[j-1]][route[j]]
			
		
		
		#total fitness and max fitness
		return max(fitnesses), sum(fitnesses)
	
	#Crossover
	def crossover(self, other):
		choice = random.choice([1, 2, 3])
		if choice == 1:
			return self.OX(other)
		elif choice == 2:
			return self.ERX(other)
		else:
			return self.AEX(other)

	# Order cross over
	def OX(self, other):
		mom_chromosome = self.chromosome
		dad_chromosome = other.chromosome


		a = random.randint(1, self.n-1)
		b = random.randint(1, self.n-1)

		if a > b:
			a, b = b, a
		
		middle_chromosome: list = dad_chromosome[a:b]

		temp_chromosome: list = mom_chromosome[b:] + mom_chromosome[:b]

		for gene in middle_chromosome:
			temp_chromosome.remove(gene)

		child_chromosome = temp_chromosome[self.n-b:] + middle_chromosome + temp_chromosome[:self.n-b]
		
		return child_chromosome
	
	#Edge recombination crossover
	def ERX(self, other):
		mom_chromosome = self.chromosome
		dad_chromosome = other.chromosome

		def find_neighbor(gene):
			neighbors = []
			mom_index = mom_chromosome.index(gene)
			dad_index = dad_chromosome.index(gene)

			if mom_index == 0:
				neighbors += [mom_chromosome[-1], mom_chromosome[1]]

			elif mom_index == self.n-1:
				neighbors += [mom_chromosome[0], mom_chromosome[self.n-2]]
			
			else:
				neighbors += [mom_chromosome[mom_index-1], mom_chromosome[mom_index+1]]
			
			if dad_index == 0:
				neighbors += [dad_chromosome[-1], dad_chromosome[1]]

			elif dad_index == self.n-1:
				neighbors += [dad_chromosome[0], dad_chromosome[self.n-2]]
			
			else:
				neighbors += [dad_chromosome[dad_index-1], dad_chromosome[dad_index+1]]
			
			return list(set(neighbors))

		neighbors = {i: find_neighbor(i) for i in range(-self.K+2, self.N+1)}
		
		index = 0
		child_chromosome = []
		gene = mom_chromosome[0]
		while len(child_chromosome) < self.n:
			child_chromosome.append(gene)

			neighs = neighbors[gene]

			neighs.sort(key= lambda x: len(neighbors[x]))

			check = False
			for g in neighs:
				if g not in child_chromosome:
					gene = g
					check = True
					break
			
			if not check:
				for g in mom_chromosome:
					if g not in child_chromosome:
						gene = g
						break

		return child_chromosome

	# Alternating Edges cross over
	def AEX(self, other):
		mom_chromosome: list =  self.chromosome
		dad_chromosome: list = other.chromosome
		child_chromosome = []

		visited = {i: False for i in range(-self.K+1, self.N+1)}

		visited[-self.K+1] = True

		current = mom_chromosome

		index = 0

		while len(child_chromosome) < self.n:
			gene = current[index]
			child_chromosome.append(gene)
			visited[gene] = True

			if current == mom_chromosome:
				current = dad_chromosome
			elif current == dad_chromosome:
				current = mom_chromosome
			
			index = current.index(gene) + 1

			if index >= self.n:
				for i in range(-self.K+1, self.N+1):
					if visited[i] == False:
						index = current.index(i)
						break
			
			elif visited[current[index]] == True:
				for i in range(-self.K+1, self.N+1):
					if visited[i] == False:
						index = current.index(i)
						break
		
		return child_chromosome
	
	#mutation
	def mutation(self, rate):
		if random.random() < rate:
			#choose mutation type
			choice = random.choice([1,2,3])
			if choice == 1:
				self.swap_mutation()
			elif choice == 2:
				self.scramble_mutation()
			else:
				self.inversion_mutation()

	#take 2 gene and swap
	def swap_mutation(self):
		a = random.randint(0, self.n-1)
		b = random.randint(0, self.n-1)

		self.chromosome[a], self.chromosome[b] = self.chromosome[b], self.chromosome[a]
	
	# take genes from a to b and scramble
	def scramble_mutation(self):
		a = random.randint(0, self.n-1)
		b = random.randint(0, self.n-1)
		if a  > b:
			a, b = b, a

		temp = self.chromosome[a:b].copy()
		random.shuffle(temp)
		
		self.chromosome = self.chromosome[:a] + temp + self.chromosome[b:]
	
	# take genes from a to b and inverse
	def inversion_mutation(self):
		a = random.randint(0, self.n-1)
		b = random.randint(0, self.n-1)
		if a > b:
			a, b = b, a

		temp = self.chromosome[a:b].copy()
		temp.reverse()
		
		self.chromosome = self.chromosome[:a] + temp + self.chromosome[b:]

#Solution class	
class GA:
	def __init__(self, N, K, distance_matrix, n, generations, mutation_rate, greedy_chromosome):
		#Populations contain n Individual
		self.populations: list[Individual] = [Individual(N, K, distance_matrix, greedy_chromosome) for _ in range(n)]
		self.n = n
		self.generations = generations
		self.distance_matrix = distance_matrix
		self.mutation_rate = mutation_rate
		self.N = N
		self.K = K
	
	def solve(self):

		self.calc_fitness()

		# note that populations[-1] is the best individual

		#initial best solution is populations[-1]
		self.best_sol = self.populations[-1]

		iteration = 0
		max_iteration = 50

		#run for ... generations
		for generation in range(self.generations):
			iteration += 1
			Probs = self.calc_fitness()

			# if populations[-1] fitness is smaller than best sol, update best sol
			if   self.populations[-1].fitness < self.best_sol.fitness:
				self.best_sol = self.populations[-1]

				iteration = 0

			
			# early stopping
			if iteration > max_iteration:
				break

			new_gen = []

			# create new popultions
			for i in range(self.n):

				# choose parent
				parent: list[Individual] = self.natural_selection(Probs)

				# cross over
				child_chromosome = parent[0].crossover(parent[1])

				child = Individual(self.N, self.K, self.distance_matrix, child_chromosome)

				#mutation
				child.mutation(self.mutation_rate)

				#add new gen
				new_gen.append(child)
		
			self.populations = new_gen
		
		
	
	#calc populations fitness
	def calc_fitness(self):
		#calc fitness for each individual
		for indiviudal in self.populations:
			indiviudal.fitness = indiviudal.calc_fitness()

		#sort in decreasing order of fitness, best fitness at lass
		self.populations.sort(reverse=True, key= lambda x: x.fitness)

		#rank selection
		sp = 1.2
		Probs = [1/self.n * (sp - (2*sp-2)*(i-1)/(self.n-1)) for i in range(1, self.n+1)]
		Probs.reverse()

		for i, individual in enumerate(self.populations):
			individual.prob = Probs[i]
		
		for i in range(1, len(Probs)):
			Probs[i] += Probs[i-1]
			
		return Probs

	
	#choose 2 parents base on rank
	def natural_selection(self, Probs):
		
		parent = []
		Probs = [0] + Probs

		for i in range(2):
			choice = random.uniform(0, Probs[-1])

			for i in range(1, len(Probs)):
				if Probs[i-1] <= choice <= Probs[i]:
					parent.append(self.populations[i-1])
					break

		return parent

	#printing solution
	def print_sol(self):
		print(self.K)
		max_cost = 0
		routes_cost = [0 for _ in range(self.K)]

		# Calculate routes' individual costs
		for truck in range(self.K):
			route_cost = 0
			for j in range(1, len(self.best_sol.Routes[truck])):
				route_cost += self.distance_matrix[self.best_sol.Routes[truck][j-1]][self.best_sol.Routes[truck][j]]
			routes_cost[truck] = route_cost
			max_cost = max(max_cost, route_cost)
			
			# Print route details
			print(len(self.best_sol.Routes[truck]))
			print(*self.best_sol.Routes[truck])
		
		print('Max Cost', max_cost)
		

	#exporting solution
	def export_sol(self, file):
		with open(file, 'w') as f:
			f.write(str(self.K) + "\n")

			for truck in range(self.K):
				f.write(str(len(self.best_sol.Routes[truck])) + "\n")

				for node in self.best_sol.Routes[truck]:
					f.write(str(node) + " ")
				
				f.write("\n")


def create_greedy_chromosome(N, K, distance_matrix):
	class Truck:
		def __init__(self):
			self.route = [0]  # Bắt đầu từ depot
			self.cost = 0

	class GreedyConstructor:
		def __init__(self, N, K, distance_matrix):
			self.N = N
			self.K = K
			self.distance_matrix = distance_matrix
			self.trucks = [Truck() for _ in range(K)]
			self.reqs = list(range(1, N + 1))
			self.combinations = [None] * K

		def greedy(self):
			chromosome = []
			while self.reqs:
				# Lấy sự kết hợp chèn tốt nhất
				combination = self.best_insert_combination()
				truck_idx = combination['truck_idx']
				route_idx = combination['idx']
				req = combination['req']
				cost = combination['cost']
				
				# Chèn yêu cầu vào route của truck
				self.trucks[truck_idx].route.insert(route_idx, req)
				# Loại bỏ yêu cầu khỏi danh sách các yêu cầu chưa phục vụ
				self.reqs.remove(req)
				
				# Đánh dấu các truck và yêu cầu đã được xử lý
				for i in range(self.K):
					if self.combinations[i] is None or self.combinations[i]['req'] == req or self.combinations[i]['truck_idx'] == truck_idx:
						self.combinations[i] = None
				
				# Cập nhật chi phí của truck
				self.trucks[truck_idx].cost = cost
			
			# Tạo chromosome từ các route của trucks
			for truck_idx, truck in enumerate(self.trucks):
				if truck_idx > 0:
					chromosome.append(-truck_idx + 1)  # Đánh dấu chuyển truck
				chromosome.extend(truck.route[1:])  # Bỏ qua depot (node 0)
			
			return chromosome

		def best_insert_combination(self):
			# Lặp qua từng truck
			for i in range(self.K):
				# Kiểm tra nếu chưa có sự kết hợp cho truck này
				if self.combinations[i] is None:
					min_cost = float('inf')
					results = []

					# Lặp qua các vị trí có thể chèn trong route của truck
					for j in range(1, len(self.trucks[i].route) + 1):
						# Tìm yêu cầu có chi phí chèn tối thiểu
						req = min(self.reqs, key=lambda x: self.insert_cost(i, j, x))
						current_cost = self.insert_cost(i, j, req)

						if current_cost == min_cost:
							results.append({
								'req': req,
								'idx': j,
							})
						if current_cost < min_cost:
							min_cost = current_cost
							results = [{
								'req': req,
								'idx': j,
							}]

					# Chọn ngẫu nhiên một trong các lựa chọn tốt nhất
					result = random.choice(results)
					route_idx = result['idx']
					req = result['req']
					min_cost = self.insert_cost(i, route_idx, req)

					# Tạo sự kết hợp
					combination = {
						'req': req,
						'truck_idx': i,
						'idx': route_idx,
						'cost': min_cost
					}
					self.combinations[i] = combination

			# Tìm sự kết hợp tốt nhất
			best_combination = min(self.combinations, key=lambda x: x['cost'])
			return best_combination

		def insert_cost(self, truck_idx, route_idx, node):
			# Kiểm tra tính hợp lệ của route_idx
			if route_idx <= 0 or route_idx > len(self.trucks[truck_idx].route):
				raise ValueError("route_idx không hợp lệ")
			
			# Lấy node trước đó trong route
			prev = self.trucks[truck_idx].route[route_idx - 1]
			
			# Nếu chèn ở cuối route
			if route_idx == len(self.trucks[truck_idx].route):
				cost = self.trucks[truck_idx].cost + self.distance_matrix[prev][node]
			else:
				# Lấy node hiện tại tại vị trí chèn
				current = self.trucks[truck_idx].route[route_idx]
				# Cập nhật chi phí bằng cách loại bỏ khoảng cách cũ và thêm khoảng cách mới
				cost = self.trucks[truck_idx].cost - self.distance_matrix[prev][current] + \
					self.distance_matrix[prev][node] + self.distance_matrix[node][current]
			
			return cost

	# Khởi tạo và chạy giải thuật greedy
	greedy_constructor = GreedyConstructor(N, K, distance_matrix)
	return greedy_constructor.greedy()


def read_input():
	# Đọc giá trị N và K
	N, K = map(int, input().split())
	
	# Đọc ma trận khoảng cách
	distance_matrix = []
	for _ in range(N + 1):
		distance_matrix.append(list(map(int, input().split())))
	return N, K, distance_matrix

def main():
	N, K, distance_matrix = read_input()
	populations_num = 100
	generations = 100
	mutation_rate = 0.1

	# Initialize a greedy Individual
	greedy_chromosome = create_greedy_chromosome(N,K,distance_matrix)

	sol = GA(N, K, distance_matrix, populations_num, generations, mutation_rate, greedy_chromosome)

	sol.solve()

	sol.print_sol()
	# print(greedy_chromosome)
if __name__ == "__main__":
	main()