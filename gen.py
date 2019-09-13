#!/usr/bin/env python
'''Contains methods to run executables and do basic general stuff'''

import os
import shutil
import tempfile
from datetime import datetime
import subprocess


def AddDirSeparator(dir):
	'''Adds a / to a string if it does not end in one'''
	if (len(dir)==0):
		raise Exception("Path was empty. Provide an absolute path.")
	if dir[-1] != '/':
		return dir + '/'
	return dir


def DeleteDir(dir):
	'''Deletes a directory if it exists'''
	if os.path.isdir(dir):
		shutil.rmtree(dir)


def Delete(path):
	'''Deletes a file or directory if it exists'''
	if os.path.isdir(path):
		DeleteDir(path)
	elif os.path.exists(path):
		os.remove(path)


def GetTempfileName(suffix):
	'''Returns the name of a file that can be saved to in the temp folder. You must delete the file manually once used.
	
	Args:
		suffix: the suffix with or without a preceding fullstop
	
	'''

	if len(suffix) > 0 and not (suffix[0] == "."):
		suffix = "." + suffix

	return os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + suffix)


def MakeDirectory(dir):
	'''Creates a directory if it does not exist'''
	if not os.path.exists(dir):
		os.mkdir(dir)
	

def Int2Str(num):
	'''Returns a number as a string without decimal places. Checks input % 1 == 0'''
	if num % 1 > 0:
		raise Exception(str(num) + " is not an integer")

	return "{:.0f}".format(num)


def TimeRun(commandArrayOrString):
	'''Runs an executable and returns how long it took'''
	start = datetime.now()
	
	Run(commandArrayOrString)
	
	end = datetime.now()
	diff = end-start
	return diff


def RunIfLastArgDoesNotExist(commandArray, printCommands=True):
	'''Runs a command if the last argument (a path) does not exist'''
	if not os.path.exists(commandArray[-1]):
		Run(commandArray,printCommands)
	

def Run(commandArrayOrString, printCommands=True):
	'''Runs a command in bash
	
	Arguments:
		commandArrayOrString:	The command to run, either as a string or as an array of strings each to be separated by a space
		printCommands:			Whether to print to the commandline that which is being executed
	'''
	if isinstance(commandArrayOrString, str):
		commandAsStr = commandArrayOrString
	else:
		commandArrayOrString = [str(i) for i in commandArrayOrString] #convert each element to a string
		commandAsStr=" ".join(commandArrayOrString)

	if printCommands:
		print('---------------------------')
		print('executing command: ' + commandAsStr)
		print('')
	
	try:
		subprocess.check_call(commandAsStr,shell=True,executable="/bin/bash")
	except RuntimeError:
		raise Exception("Error executing:\n" + commandAsStr)

	if printCommands:
		print('Command Complete: ' + commandAsStr)	
		print('')
		print('---------------------------')


def Run_CaptureOutput(commandArrayOrString, printCommands=True):
	'''Runs a command and returns that which was output to the commandline
	
	Arguments:
		commandArrayOrString:	The command to run, either as a string or as an array of strings each to be separated by a space
		printCommands:			Whether to print to the commandline that which is being executed
	'''
	if isinstance(commandArrayOrString, str):
		commandAsStr = commandArrayOrString
	else:
		commandArrayOrString = [str(i) for i in commandArrayOrString] #convert each element to a string
		commandAsStr=" ".join(commandArrayOrString)
	
	if printCommands:
		print('---------------------------')
		print('executing command: ' + commandAsStr)
	
	
	
	
	if os.name == 'posix':
		proc=subprocess.Popen(commandAsStr, stdout=subprocess.PIPE,shell=True, executable="/bin/bash")
	else:
		proc=subprocess.Popen(commandAsStr, stdout=subprocess.PIPE,shell=True)
		
	#run and get the output
	output = str(proc.communicate()[0])
	
	#Swap things like \t for actual tab etc
	output=output.replace('\\t','\t').replace('\\n','\n')
	
	#Strip the preceding b' and final '
	output = output[2:-1]
	
	#Check it exit OK
	if not proc.returncode == 0:
		raise Exception("Error executing:\n" + commandAsStr +"\noutput:\n" + output)
	
	if printCommands:
		print('---------------------------')
		print('Command Complete: ' + commandAsStr)	
	
	return output
