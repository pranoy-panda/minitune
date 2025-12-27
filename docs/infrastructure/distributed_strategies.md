# Distributed Strategies: Under the Hood

Before using `minitune`'s high-level trainers, it is useful to understand what is happening at the tensor level. 

This section breaks down **Native DDP**, **Gradient Accumulation**, and **Memory Optimizations** as they exist in raw PyTorch.

## 1. The Raw DDP Loop
In a raw implementation, we are required to do the manual device placement, process group initialization, and gradient synchronization.

### The "Boilerplate" of Distributed Training
Every distributed run must start by establishing the "Communication Ring" (NCCL) and binding the process to a specific GPU.

```python

def train():
	if global_rank==0:
		initialize_services() # Wandb, etc
	
	data_loader = DataLoader(train_dataset, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True))
	model = MyModel()
	if os.path.exists('models/latest_checkpoint.pth'):
		# Load latest checkpoint, optimizer state and other vars needed to restore the training state
		model.load_state_dict(torch.load('models/latest_checkpoint.pth'))
		
	# wrap model
	model = DistributedDataParallel(model, device_ids=[local_rank])
	optimizer = torch.optim.Adam(model.parameters(), lr=10e-4,eps=1e-9)
	loss_fn = torch.nn.CrossEntropyLoss()
	
	for epoch in range(num_epochs):
		for data, labels in data_loader:
			# gradient accumulation for 100 steps
			if (step_number+1)%100 != 0 and not last_step:
				with model.no_sync(): # disable gradient synchronization
```					loss = loss_fn(model(data),labels)) # forward prop
					loss.backward() # backprop + gradient accumulation
			else:
				loss = loss_fn(model(data),labels))
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
			if global_rank==0:
				collect_stats() # send data to Wandb
		if global_rank==0:
			torch.save(model.state_dict(),"latest_checkpoint.pth")
			
if __name__=="__main__":
	local_rank = int(os.environ['LOCAL_RANK'])
	global_rank = int(os.environ['RANK'])
	
	train()
	
	destroy_process_group()
```	
			
			
