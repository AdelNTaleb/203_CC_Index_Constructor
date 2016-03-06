# CrossCutting

## Installation 

just run pip install -r requiuerement.txt

## How to use Git-Hub

https://www.youtube.com/playlist?list=PLfdtiltiRHWFEbt9V04NrbmksLV4Pdf3j

### Pre-requisite:
Install github 

### PULL - GET THE FILE 
Copy link on website
open terminal
with cd, go to the file where you want to put the file (within your computer: ex: desktop)
Enter the initial Pull
> git clone <link>
if required: enter your git user and pw.
New Folder on Desktop: CrossCutting

### CONFIGURE YOUR GIT SPACE
>git config --global user.email "you@example.com”
>git config --global user.name "Your Name”
>git config --global push.default simple
  

###Useful practice
On the terminal, run “git status”: it will display all the files that have been modified since the last pull, and show you if you have the most updated version of the project.
BEFORE MODIFYING: do a 
> git status 
and if your version is outdated, do a 
> git pull

Do a git pull regularly 


### MODIFY AS MUCH AS YOU WANT FILES WITHIN THIS FOLDER

### PUSH - STEPS TO FOLLOW TO PUT YOUR VERSION ON THE GITHUB 
Do a git add of the files you modified:
> git add <file1> <file2>
Then, to create a back-up of your modification on the previous files. (would be easily traceable), enter on the terminal 
> git commit -m "ALL YOUR COMMENTS ON WHAT YOU’VE JUST DONE
Finally, use push to put all your modification on the drive.
> git push
