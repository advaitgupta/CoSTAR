# CoSTA*: Cost-Sensitive Toolpath Agent for Multi-turn Image Editing  

---

## **Introduction**  
**CoSTA*** is a cost-sensitive toolpath agent designed to solve **multi-turn image editing** tasks efficiently. It integrates **Large Language Models (LLMs)** and **graph search algorithms** to dynamically select AI tools while balancing cost and quality. Unlike traditional **text-to-image** models (e.g., **Stable Diffusion, DALLE-3**), which struggle with complex image editing workflows, **CoSTA*** constructs an optimal **toolpath** using an **LLM-guided hierarchical planning strategy** and an **A*** search-based selection process.  


This repository provides:  
- The official **codebase** for **CoSTA***.  
- Scripts to **generate and optimize toolpaths** for multi-turn image editing.  

## **Live Demo**  
Try out **CoSTA*** online: **[Live Demo](https://storage.googleapis.com/costa-frontend/index.html)**  

---

## **Dataset**  
We provide a **benchmark dataset** with **121 images** for testing CoSTA*, containing **image-only** and **text+image** tasks.  

---

## **Features**  
✅ **Hierarchical Planning** – Uses **LLMs** to decompose a task into a **subtask tree** which is used for constructing the final **Tool Subgraph**.  
✅ **Optimized Tool Selection** – A* search is applied on the **Tool Subgraph** for **cost-efficient, high-quality** pathfinding.  
✅ **Multimodal Support** – Switches between **text** and **image modalities** for enhanced editing.  
✅ **Quality Evaluation via VLM** – Automatically assesses tool outputs to estimate the actual quality before progressing further.  
✅ **Adaptive Retry Mechanism** – If the output doesn’t meet the quality threshold, it is **retried with updated hyperparameters**.  
✅ **Balancing Cost vs. Quality** – A* search **does not just minimize cost** but also **optimizes quality**, allowing users to adjust **α (alpha)** to control cost vs. quality trade-off.  
✅ **Supports 24 AI Tools** – Integrates **YOLO, GroundingDINO, Stable Diffusion, CLIP, SAM, DALL-E**, and more.  

---

## **Installation**  
### **1. Clone the Repository**  
```bash
git clone 
cd CoSTAR  
```

### **2. Install Dependencies**  
Ensure you have **Python 3.8+** and install dependencies (most other dependencies are auto-installed when models are run):  
```bash
pip install -r requirements.txt  
```

### **3. Download Pre-trained Checkpoints**  
The required pre-trained model checkpoints must be downloaded from **Google Drive** and placed in the `checkpoints/` folder. The link to download the checkpoints is provided in `checkpoints/checkpoints.txt`.  

---

## **Usage**
*Note: The API keys for OpenAI and StabilityAI need to be set in the run.py file before executing.*
To execute **CoSTA***, run:  
```bash 
python run.py --image path/to/image.png --prompt "Edit this image" --output output.json --output_image final.png --alpha 0  
``` 

Example:  
```bash 
python run.py --image inputs/sample.jpg --prompt "Replace the cat with a dog and expand the image" --output Tree.json --output_image final_output.png --alpha 0
```  

- `--image`: Path to input image.  
- `--prompt`: Instruction for editing.  
- `--output`: Path to save generated subtask tree.  
- `--output_image`: Path to save the final output.  
- `--alpha`: Cost-quality trade-off parameter.  

---

## **Running Individual Components**  
*The **main functions** in the following scripts need to be **uncommented**, and the **paths, hyperparameters, and API keys** must be **modified** before execution.*  

### **1. Generate a Subtask Tree**  
Modify `subtask_tree.py` by providing the **input image path and prompt**, then run:  
```bash 
python subtask_tree.py  
```  

### **2. Build a Tool Subgraph**  
Modify `tool_subgraph.py` to use the generated `Tree.json`, then execute:  
```bash  
python tool_subgraph.py  
```  

### **3. Run A\* Search for Optimal Toolpath**  
Modify `astar_search.py` with updated paths and hyperparameters, then run:  
```bash  
python astar_search.py  
```  

### **4. Visualize the Process**  
A step-by-step **live example** can be found in `Demo.ipynb`, which provides an interactive **Jupyter Notebook** for understanding the workflow.  

---

## **Directory Structure**  
```bash  
CoSTAR/  
├── checkpoints/         
│   ├── checkpoints.txt  
├── configs/             
│   ├── tools.yaml       
├── inputs/             
│   ├── 40.jpeg         
├── outputs/            
│   ├── final.png       
├── prompts/           
│   ├── 40.txt          
├── requirements/       
│   ├── craft.txt       
│   ├── deblurgan.txt   
│   ├── easyocr.txt     
│   ├── google_cloud.txt
│   ├── groundingdino.txt
│   ├── magicbrush.txt  
│   ├── realesrgan.txt  
│   ├── sam.txt         
│   ├── stability.txt   
│   ├── yolo.txt        
├── results/           
│   ├── final.png       
│   ├── img1.png        
│   ├── img2.png        
│   ├── img3.png        
│   ├── img4.png        
│   ├── img5.png        
├── tools/              
│   ├── dalleimage.py  
│   ├── groundingdino.py  
│   ├── sam.py  
│   ├── stabilityoutpaint.py  
│   ├── yolov7.py  
│   └── ...  
├── .gitignore          
├── LICENSE           
├── README.md       
├── Demo.ipynb       
├── run.py             
├── subtask_tree.py   
├── tool_subgraph.py  
├── astar_search.py    
```  
