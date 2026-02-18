# Requirements

- Python 3.9 or higher
- python-dotenv — loads .env files
- cryptography — optional HMAC support (stdlib hmac also works)
- Everything else uses the Python standard library

# Start

- Double-click setup.bat
- Open .env in Notepad, add your ANTHROPIC_API_KEY
- python main.py

# Usage

          python main.py                                      
          python main.py --games 5                           
          python main.py --name1 Kira --class1 Rogue         
          python main.py --name2 Vorn --class2 Berserker
          python main.py --id1 abc123def --id2 xyz456uvw     
          python main.py --thinking                           
          python main.py --thinking1                          
          python main.py --quiet --games 20                  
          python main.py --status                             
          python main.py --data-dir                          
          python main.py --help

# AI Techniques

- APE (Automatic Prompt Engineering): Each agent maintains a pool of competing system prompts. After every N games, the worst-performing prompts are replaced by LLM-generated variants based on the best current prompt. Selection uses UCB1 so unexplored variants get evaluated before being judged. Agents literally evolve their own voice and strategy over time.
  
- UCB1 Multi-Armed Bandit: Every ***(action, reward)*** pair is tracked per agent. Before each decision, the agent's context includes its historically best action, balancing exploitation of proven tactics with exploration of underused ones. Fallback decisions (when rate-limited) also use UCB1.
  
- Episodic Semantic Memory: Before each turn, the agent runs a deterministic hash-based embedding of the current battle state, retrieves semantically similar past episodes, and injects natural language into the decision context. 
