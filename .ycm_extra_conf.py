import os
import re
from typing import List

# Base flags
flags = [
    '-x',
    'c++',
    '-Wall',
    '-Wextra',
    '-Werror',
]


# Exclude directories with specific name
EXCLUDE_DIRS = [
    "vs2008",
    ".git"
]


def ReadFile(file: str) -> List[str]:
    with open(file, 'r') as f:
        data = f.read().split("\n")
    return data


def ParsePremake5File(text: List[str]) -> List[str]:
    result: List[str] = []
    tmpContent: str = ""

    text_iter = iter(text)
    for line in text_iter:
        if "includedirs" in line:
            tmpContent += line
            while "}" not in line:
                line = next(text_iter)
                tmpContent += line
            tmpContent = [str(directory.replace("\"", "").strip()) for directory in re.sub(r'^.*?{', '', tmpContent.replace("}", "")).split(",")]
            result.extend(tmpContent)
            tmpContent = ""
    return result


def ExtractIncludeDirs(premake_file: str) -> List[str]:
    text: List[str] = ReadFile(premake_file)
    included_dirs = ParsePremake5File(text)
    return included_dirs


def GetIncludesFromPremake(project_dir: str) -> List[str]:
    premake_file: str = os.path.join(project_dir, "premake5.lua")
    if os.path.exists(premake_file):
        premake_included_dirs = ExtractIncludeDirs(premake_file)
        premake_included_dirs = [dirs.replace("./", project_dir+"/") for dirs in premake_included_dirs]
        return premake_included_dirs


def MoveDirectoryUpUntil(directory, list_required=[]) -> str:
    required = list_required.copy()
    current_content = os.listdir(directory)

    for current_object in current_content:
        if current_object in required: required.remove(current_object)

    if required != []:
        directory_up = os.path.split(directory)[0]
        if (directory_up in os.environ["HOME"]) | (".git" in current_content): # i.e. /home/ or highest up in project
            print(f"WARNING: No directory found containing: {list_required}")
            return None
        directory = MoveDirectoryUpUntil(directory_up, list_required)
    return directory


def GetProjectDir(directory):
    project_dir = MoveDirectoryUpUntil(directory, ['.git'])
    return project_dir


def AddAllDirectories(flags, working_directory):
    proj_dir = GetProjectDir(working_directory)

    folders = GetIncludesFromPremake(proj_dir)
    if folders != []:
        for fold in folders:
            flags.append(f'-I{fold}')

    return flags


# Get the relevant directories and add the flags
working_directory = os.path.abspath('.')
flags = AddAllDirectories(flags, working_directory)

def Settings( **kwargs ):
    return {'flags': flags}
