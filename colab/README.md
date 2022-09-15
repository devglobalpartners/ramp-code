# Train RAMP Building Detection model on your custom dataset

#### 0. Download the ramp codebase onto your local computer. 

It doesn't matter where it is downloaded. You will only need to do this in order to upload the *colab/jupyterlab_on_colab.ipynb* file to Google Colab in the next step.

#### 1. Start a Jupyter Lab instance on Colab

Go to the Google Colab website. 

*Important note*: make sure that you are logged into Colab as the same user whose Google Drive you will be mounting for permnanent storage of your training data! You will get a mysterious error if you try to mount the Google Drive from a different 'Google persona' than the one you used to log in to Colab. 

On the menu at the upper left, select 'File > Upload Notebook', and upload the 'jupyterlab_on_colab.ipynb' notebook file from the 'colab' directory of the ramp codebase. 

Execute the commands in that notebook within Colab to start your jupyter lab session, and then come back to this Readme file to finish your ramp setup on Colab.

#### 2. Create the "RAMP_HOME" directory.

Assuming you now have a running Jupyter Lab session going, your personal Google Drive will be in the directory list in the window at left. Navigate to */drive/MyDrive*; that is your Google Drive home directory. 

Once you're in this directory, click on the 'Create New Folder' icon at the top of the file menu, and name your new folder "RAMP_HOME". You will only need to do this once, as anything written here will remain on your Personal Google Drive until you remove it.


#### 3. Navigate to *RAMP_HOME*, and create another new folder named "ramp-data".

#### 4. In the main panel of Jupyter Lab, there are a number of applications you can select. Click on the Terminal application to open a terminal into the remote host. This terminal is running Linux, and will usually place you inside a directory named '/content'. You'll want to navigate to your *RAMP_HOME* directory. 

To do this, type 'ls' at the command line. This will list all the subdirectories of '/content', one of which will be 'drive'. Type:

```
cd drive/MyDrive
```

to get back to your main Personal Drive directory, and then type 

```
cd RAMP_HOME
```

to be placed in the *RAMP_HOME* directory. From here, you can download and install the ramp codebase. 

#### 5. Clone the ramp github repository.

The next step is to authenticate yourself to github, so it will allow you to download the ramp code. Unfortunately, authenticating yourself to github is no longer as easy as entering your name and password, but it's not too much harder. In order to authenticate, you'll first need to generate a "Personal Access Token" from your account on the github webside. Instructions for doing this are at the end of this README. 

Note that if you don't already have a github account, you'll need to create one before creating a personal access token!

Once you have a personal access token, run the following commands from the terminal inside JupyterLab:

```
git config --global user.name "your_github_username"
git config --global user.email "your_email"
git clone https://github.com/carolynpjohnston/ramp-staging.git
```
You'll be prompted for your username and password. For your password, enter that Personal Access Token that you saved. (Note that you can copy-paste this from your saved location into the browser).

The ramp code will then be downloaded to your RAMP_HOME directory.

#### 6. Install the required packages & binaries to run the ramp code

The required packages are specified in a Makefile in the colab directory within the ramp codebase. Type the following at the Jupyter Lab command line:

```
cd ramp-staging/colab
make install
```

This process will take quite a while to complete, as the ramp and solaris modules have many dependencies.

#### 7. Install the ramp code. 

After 'make install' finishes, you will still be in the 'ramp-staging/colab' directory. Typing `cd ..` will step you up one level in the directory structure, so you'll be in the 'ramp-staging' directory. Type the following at the command line:

```
cd ..
pip install -e .
```

to install the ramp modules. *Be sure to include the period at the end of the pip install command!* That period tells the pip command to look for the information it needs in the current working directory. The *pip install* command will finish quickly.

To test whether your installation worked, start a python session as follows:

```
python
>> import ramp
```

If *import ramp* runs successfully, you have installed the ramp modules. Note that *import ramp* has run successfully if you see the following output and there are no error messages:

```
python
>> import ramp
>>
```

#### 8. Run the Colab model training notebook: `train_ramp_model_on_colab.ipynb`.
