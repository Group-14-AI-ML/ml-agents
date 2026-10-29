# **PushBlock manual training and result saving without active Unity Editor** 

## **1. Prerequisites & Environment Setup** 

Activate your Python 3.10 virtual environment and install the required telemetry extraction dependencies. 

### **Linux / macOS** 

_source <your_virt_env_name>/bin/activate_ 

### **Windows (PowerShell / CMD)** 

_<your_virt_env_name>\Scripts\Activate.ps1 # or in Command Prompt:_ 

_<your_virt_env_name>\Scripts\activate._ bat 

## **2. Compiling the Standalone Build in Unity** 

Building a standalone binary eliminates the need to manually open the Unity Editor or press Play. 

1. Open the project in **Unity Hub** . 

2. Open the scene: Assets/ML-Agents/Examples/PushBlock/Scenes/PushBlock. 

3. In the top menu, go to **File > Build Settings...** 

4. Confirm that **PushBlock** is checked under **Scenes in Build** (click _Add Open Scenes_ if it is missing). 

5. Select **PC, Mac & Linux Standalone** and set the target platform: 

   - **Linux:** Target Platform: Linux, Architecture: x86_64. 

   - **Windows:** Target Platform: Windows, Architecture: x86_64. 

   - **macOS:** Target Platform: macOS, Architecture: Apple Silicon or Intel x86_64. 

6. Click **Build** (do not click _Build and Run_ ). 

7. Save the build directly into the repository Project/ directory with the filename **PushBlock** . 

## **3. File Placement & Platform Permissions** 

Ensure all compiled build artifacts reside together inside the Project/ folder. 

### **Linux** 

- **Files in Project/:** PushBlock.x86_64, UnityPlayer.so, and the PushBlock_Data/ directory. 

- **Grant execution permissions:** _chmod +x Project/PushBlock.x86_64_ 

### **Windows** 

- **Files in Project/:** PushBlock.exe, UnityPlayer.dll, and the PushBlock_Data\ folder. 

- Ensure Windows Defender / antivirus software does not block the newly generated executable. 

### **macOS** 

- **Files in Project/:** PushBlock.app. 

- **Grant permissions and bypass Gatekeeper quarantine:** _chmod +x Project/PushBlock.app/Contents/MacOS/*_ 

## **4. Setting Up the Telemetry Script** 

Place **extract_telemetry.py** in the root repository directory (ml-agents/), adjacent to config/ and Project/. **Directory layout:** 

ml-agents/ 

├── config/ppo/PushBlock.yaml 

- ├── extract_telemetry.py 

- ├── Project/ 

- │   ├── PushBlock.x86_64 (or .exe / .app) 

- │   ├── UnityPlayer.so (or .dll) 

- │   └── PushBlock_Data/ 

- └── results/ 

## **5. Running a Headless Test Training Session** 

Run mlagents-learn pointing to the standalone binary via --env. The simulation executes and terminates automatically in the background. 

### **Linux** 

_mlagents-learn config/ppo/PushBlock.yaml --run-id=test_run_to_delete --env=Project/PushBlock.x86_64 --no-graphics --force_ 

### **Windows** 

_mlagents-learn config\ppo\PushBlock.yaml --run-id=test_run_to_delete --env=Project\PushBlock.exe --no-graphics --force_ 

### **macOS** 

_mlagents-learn config/ppo/PushBlock.yaml --run-id=test_run_to_delete --env=Project/PushBlock.app --no-graphics --force_ 

#### **Key Flags Explained:** 

- --env: Connects directly to the compiled game binary. 

- --no-graphics: Disables 3D window rendering to free maximum CPU/GPU compute power for PPO training. 

- --force: Overwrites any previous runs sharing the same run-id. 

## **6. Extracting Telemetry into CSV** 

After training concludes and exports the .onnx model, run extract_telemetry.py passing the run-id and the output CSV filename: 

_python extract_telemetry.py test_run_to_delete dataset_test.csv_ 

## **7. Verifying the CSV Output** 

Inspect the generated CSV file to confirm that all 15 columns have been populated correctly by opening with a .csv viewer 

