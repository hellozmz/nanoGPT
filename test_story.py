### load dataset
from datasets import load_dataset

# 加载 ROCStories 数据集
dataset = load_dataset("story_cloze", data_dir="data/story_cloze")

### preprocess data
from transformers import GPT2Tokenizer

# 加载 GPT-2 分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 定义预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# 预处理数据
tokenized_dataset = dataset.map(preprocess_function, batched=True)

### fine-tuning
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer

# 加载 GPT-2 预训练模型
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-simple-story",
    num_train_epochs=3,               # 训练的轮数
    per_device_train_batch_size=4,    # 每个设备的批量大小
    save_steps=10_000,                # 保存模型的步数
    save_total_limit=2,               # 只保留最新的两个模型
    logging_dir='./logs',             # 日志存储目录
    logging_steps=500,                # 记录日志的步数
)

# 使用 Trainer 进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],  # 训练数据
    eval_dataset=tokenized_dataset['validation'], # 验证数据
)

### start training
trainer.train()

# 保存模型
trainer.save_model("./gpt2-simple-story")

### generate story
# 加载训练好的模型
model = GPT2LMHeadModel.from_pretrained('./gpt2-simple-story')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 生成故事
def generate_story(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1, temperature=0.7, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 给定提示生成简单故事
story = generate_story("Once upon a time in a small village")
print(story)

