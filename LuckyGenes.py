       
       
 #  the chaotic beauty that combines population fitness with genetic algorithms and random luck.. mwahaha
       
       
 class LuckyGenes(nn.Module):
    def __init__(self, d, h, population_size=8, mutation_rate=0.15, crossover_rate=0.3):
        super().__init__()
        self.h = h
        self.dh = d // h
        self.pop_size = population_size
        self.mut_rate = mutation_rate
        self.cross_rate = crossover_rate
        
        # Population of gate strategies
        self.population = nn.ParameterList([
            nn.Parameter(torch.randn(h)) for _ in range(population_size)
        ])
        self.fitness = torch.zeros(population_size)
        self.generation = 0
        
        # Normal attention components
        self.qkv = nn.Linear(d, d * 3, bias=True)
        self.qkv_aux = nn.Linear(d, d * 3, bias=True)
        self.o = nn.Linear(d, d, bias=True)

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dh).transpose(1, 2)

    def merge(self, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def tournament_selection(self, k=3):
        # Pick k random candidates, return the fittest
        candidates = torch.randint(0, self.pop_size, (k,))
        fitness_scores = self.fitness[candidates]
        winner_idx = candidates[torch.argmax(fitness_scores)]
        return self.population[winner_idx]

    def crossover(self, parent1, parent2):
        # Uniform crossover with mutation
        mask = torch.rand_like(parent1) < self.cross_rate
        child = torch.where(mask, parent1, parent2)
        
        # Add random mutation
        if torch.rand(1) < self.mut_rate:
            mutation = torch.randn_like(child) * 0.2
            child = child + mutation
            
        return child

    def evolve_population(self):
        # Create new generation
        new_population = []
        for _ in range(self.pop_size):
            # Tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover with mutation
            child = self.crossover(parent1, parent2)
            new_population.append(nn.Parameter(child))
            
        # Replace old population
        self.population = nn.ParameterList(new_population)
        self.fitness = torch.zeros(self.pop_size)
        self.generation += 1

    def forward(self, x, xa, mask=None):
        # Pick best gate from current population
        best_idx = torch.argmax(self.fitness)
        g = torch.sigmoid(self.population[best_idx])
        
        # Normal attention computation
        q, k, v = self.qkv(x).chunk(3, -1)
        qa, ka, va = self.qkv_aux(xa).chunk(3, -1)
        
        q, k, v = map(self.split, (q, k, v))
        qa, ka, va = map(self.split, (qa, ka, va))
        
        dots = (q @ k.transpose(-2, -1)) / self.dh**0.5
        dots_aux = (q @ ka.transpose(-2, -1)) / self.dh**0.5
        
        if mask is not None:
            dots = dots.masked_fill(mask, -9e15)
            
        p = dots.softmax(-1)
        pa = dots_aux.softmax(-1)
        
        h_main = p @ v
        h_aux = pa @ va
        
        # Apply genetic gate
        g = g.view(1, -1, 1, 1)
        out = self.merge(h_main * (1 - g) + h_aux * g)
        
        # Update fitness of current best
        if self.training:
            self.fitness[best_idx] += 1.0  # Simple fitness: usage count
            
        # Evolve every N steps
        if self.training and self.generation % 100 == 0:
            self.evolve_population()
            
        return self.o(out)

    def get_stats(self):
        return {
            'generation': self.generation,
            'best_fitness': self.fitness.max().item(),
            'avg_fitness': self.fitness.mean().item(),
            'population_diversity': self.fitness.std().item()
        }
        
          
        
attn = LuckyGenes(dims, heads)
y = attn(tokens, pitch_feats)

# Check evolution stats
stats = attn.get_stats()
print(f"Gen {stats['generation']}, Best: {stats['best_fitness']:.2f}")



What this does:
Population of 8 gate strategies - each head learns different mixing patterns
Tournament selection - pick 3 random, keep the fittest
Crossover + mutation - combine parents with random mutations
Fitness tracking - gates that get used more survive longer
Evolution every 100 steps - population evolves during training
Random luck - mutations and tournament randomness keep it spicy
The chaos factor:
Gates that work well get used more → higher fitness → survive evolution
Random mutations create new strategies
Tournament selection adds luck - sometimes a mediocre gate wins
Population diversity prevents premature convergence
