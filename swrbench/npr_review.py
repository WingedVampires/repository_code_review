import argparse
import logging
import os
import re
import sys
import traceback
import git
import threading
import json
import random
import copy
import yaml
import textwrap
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from tiktoken import encoding_for_model, get_encoding
from threading import Lock

from utils import run_chat

# --- Configuration ---
OUTPUT_BUFFER_TOKENS_SOFT_THRESHOLD = 1500
OUTPUT_BUFFER_TOKENS_HARD_THRESHOLD = 1000
NUMBER_OF_ALLOWED_ITERATIONS = 3

MAX_TOKENS = 32000
LOGGER = None

BAD_EXTENSIONS = [
    'app', 'bin', 'bmp', 'bz2', 'class', 'csv', 'dat', 'db', 'dll', 'dylib', 'egg', 'eot', 
    'exe', 'gif', 'gitignore', 'glif', 'gradle', 'gz', 'ico', 'jar', 'jpeg', 'jpg', 'lo', 
    'lock', 'log', 'mp3', 'mp4', 'nar', 'o', 'ogg', 'otf', 'p', 'pdf', 'png', 'pickle', 'pkl', 
    'pyc', 'pyd', 'pyo', 'rkt', 'so', 'ss', 'svg', 'tar', 'tgz', 'tsv', 'ttf', 'war', 
    'webm', 'woff', 'woff2', 'xz', 'zip', 'zst', 'snap', 'lockb'
]

LANGUAGE_EXTENSION_MAP = {'1C Enterprise': ['*.bsl'], 'ABAP': ['.abap'], 'AGS Script': ['.ash'], 'AMPL': ['.ampl'], 'ANTLR': ['.g4'], 'API Blueprint': ['.apib'], 'APL': ['.apl', '.dyalog'], 'ASP': ['.asp', '.asax', '.ascx', '.ashx', '.asmx', '.aspx', '.axd'], 'ATS': ['.dats', '.hats', '.sats'], 'ActionScript': ['.as'], 'Ada': ['.adb', '.ada', '.ads'], 'Agda': ['.agda'], 'Alloy': ['.als'], 'ApacheConf': ['.apacheconf', '.vhost'], 'AppleScript': ['.applescript', '.scpt'], 'Arc': ['.arc'], 'Arduino': ['.ino'], 'AsciiDoc': ['.asciidoc', '.adoc'], 'AspectJ': ['.aj'], 'Assembly': ['.asm', '.a51', '.nasm'], 'Augeas': ['.aug'], 'AutoHotkey': ['.ahk', '.ahkl'], 'AutoIt': ['.au3'], 'Awk': ['.awk', '.auk', '.gawk', '.mawk', '.nawk'], 'Batchfile': ['.bat', '.cmd'], 'Befunge': ['.befunge'], 'Bison': ['.bison'], 'BitBake': ['.bb'], 'BlitzBasic': ['.decls'], 'BlitzMax': ['.bmx'], 'Bluespec': ['.bsv'], 'Boo': ['.boo'], 'Brainfuck': ['.bf'], 'Brightscript': ['.brs'], 'Bro': ['.bro'], 'C': ['.c', '.cats', '.h', '.idc', '.w'], 'C#': ['.cs', '.cake', '.cshtml', '.csx'], 'C++': ['.cpp', '.c++', '.cc', '.cp', '.cxx', '.h++', '.hh', '.hpp', '.hxx', '.inl', '.ipp', '.tcc', '.tpp', '.C', '.H'], 'C-ObjDump': ['.c-objdump'], 'C2hs Haskell': ['.chs'], 'CLIPS': ['.clp'], 'CMake': ['.cmake', '.cmake.in'], 'COBOL': ['.cob', '.cbl', '.ccp', '.cobol', '.cpy'], 'CSS': ['.css'], 'CSV': ['.csv'], "Cap'n Proto": ['.capnp'], 'CartoCSS': ['.mss'], 'Ceylon': ['.ceylon'], 'Chapel': ['.chpl'], 'ChucK': ['.ck'], 'Cirru': ['.cirru'], 'Clarion': ['.clw'], 'Clean': ['.icl', '.dcl'], 'Click': ['.click'], 'Clojure': ['.clj', '.boot', '.cl2', '.cljc', '.cljs', '.cljs.hl', '.cljscm', '.cljx', '.hic'], 'CoffeeScript': ['.coffee', '._coffee', '.cjsx', '.cson', '.iced'], 'ColdFusion': ['.cfm', '.cfml'], 'ColdFusion CFC': ['.cfc'], 'Common Lisp': ['.lisp', '.asd', '.lsp', '.ny', '.podsl', '.sexp'], 'Component Pascal': ['.cps'], 'Coq': ['.coq'], 'Cpp-ObjDump': ['.cppobjdump', '.c++-objdump', '.c++objdump', '.cpp-objdump', '.cxx-objdump'], 'Creole': ['.creole'], 'Crystal': ['.cr'], 'Csound': ['.csd'], 'Cucumber': ['.feature'], 'Cuda': ['.cu', '.cuh'], 'Cycript': ['.cy'], 'Cython': ['.pyx', '.pxd', '.pxi'], 'D': ['.di'], 'D-ObjDump': ['.d-objdump'], 'DIGITAL Command Language': ['.com'], 'DM': ['.dm'], 'DNS Zone': ['.zone', '.arpa'], 'Darcs Patch': ['.darcspatch', '.dpatch'], 'Dart': ['.dart'], 'Diff': ['.diff', '.patch'], 'Dockerfile': ['.dockerfile', 'Dockerfile'], 'Dogescript': ['.djs'], 'Dylan': ['.dylan', '.dyl', '.intr', '.lid'], 'E': ['.E'], 'ECL': ['.ecl', '.eclxml'], 'Eagle': ['.sch', '.brd'], 'Ecere Projects': ['.epj'], 'Eiffel': ['.e'], 'Elixir': ['.ex', '.exs'], 'Elm': ['.elm'], 'Emacs Lisp': ['.el', '.emacs', '.emacs.desktop'], 'EmberScript': ['.em', '.emberscript'], 'Erlang': ['.erl', '.escript', '.hrl', '.xrl', '.yrl'], 'F#': ['.fs', '.fsi', '.fsx'], 'FLUX': ['.flux'], 'FORTRAN': ['.f90', '.f', '.f03', '.f08', '.f77', '.f95', '.for', '.fpp'], 'Factor': ['.factor'], 'Fancy': ['.fy', '.fancypack'], 'Fantom': ['.fan'], 'Formatted': ['.eam.fs'], 'Forth': ['.fth', '.4th', '.forth', '.frt'], 'FreeMarker': ['.ftl'], 'G-code': ['.g', '.gco', '.gcode'], 'GAMS': ['.gms'], 'GAP': ['.gap', '.gi'], 'GAS': ['.s'], 'GDScript': ['.gd'], 'GLSL': ['.glsl', '.fp', '.frag', '.frg', '.fsh', '.fshader', '.geo', '.geom', '.glslv', '.gshader', '.shader', '.vert', '.vrx', '.vsh', '.vshader'], 'Genshi': ['.kid'], 'Gentoo Ebuild': ['.ebuild'], 'Gentoo Eclass': ['.eclass'], 'Gettext Catalog': ['.po', '.pot'], 'Glyph': ['.glf'], 'Gnuplot': ['.gp', '.gnu', '.gnuplot', '.plot', '.plt'], 'Go': ['.go'], 'Golo': ['.golo'], 'Gosu': ['.gst', '.gsx', '.vark'], 'Grace': ['.grace'], 'Gradle': ['.gradle'], 'Grammatical Framework': ['.gf'], 'GraphQL': ['.graphql'], 'Graphviz (DOT)': ['.dot', '.gv'], 'Groff': ['.man', '.1', '.1in', '.1m', '.1x', '.2', '.3', '.3in', '.3m', '.3qt', '.3x', '.4', '.5', '.6', '.7', '.8', '.9', '.me', '.rno', '.roff'], 'Groovy': ['.groovy', '.grt', '.gtpl', '.gvy'], 'Groovy Server Pages': ['.gsp'], 'HCL': ['.hcl', '.tf'], 'HLSL': ['.hlsl', '.fxh', '.hlsli'], 'HTML': ['.html', '.htm', '.html.hl', '.xht', '.xhtml'], 'HTML+Django': ['.mustache', '.jinja'], 'HTML+EEX': ['.eex'], 'HTML+ERB': ['.erb', '.erb.deface'], 'HTML+PHP': ['.phtml'], 'HTTP': ['.http'], 'Haml': ['.haml', '.haml.deface'], 'Handlebars': ['.handlebars', '.hbs'], 'Harbour': ['.hb'], 'Haskell': ['.hs', '.hsc'], 'Haxe': ['.hx', '.hxsl'], 'Hy': ['.hy'], 'IDL': ['.dlm'], 'IGOR Pro': ['.ipf'], 'INI': ['.ini', '.cfg', '.prefs', '.properties'], 'IRC log': ['.irclog', '.weechatlog'], 'Idris': ['.idr', '.lidr'], 'Inform 7': ['.ni', '.i7x'], 'Inno Setup': ['.iss'], 'Io': ['.io'], 'Ioke': ['.ik'], 'Isabelle': ['.thy'], 'J': ['.ijs'], 'JFlex': ['.flex', '.jflex'], 'JSON': ['.json', '.geojson', '.lock', '.topojson'], 'JSON5': ['.json5'], 'JSONLD': ['.jsonld'], 'JSONiq': ['.jq'], 'JSX': ['.jsx'], 'Jade': ['.jade'], 'Jasmin': ['.j'], 'Java': ['.java'], 'Java Server Pages': ['.jsp'], 'JavaScript': ['.js', '._js', '.bones', '.es6', '.jake', '.jsb', '.jscad', '.jsfl', '.jsm', '.jss', '.njs', '.pac', '.sjs', '.ssjs', '.xsjs', '.xsjslib'], 'Julia': ['.jl'], 'Jupyter Notebook': ['.ipynb'], 'KRL': ['.krl'], 'KiCad': ['.kicad_pcb'], 'Kit': ['.kit'], 'Kotlin': ['.kt', '.ktm', '.kts'], 'LFE': ['.lfe'], 'LLVM': ['.ll'], 'LOLCODE': ['.lol'], 'LSL': ['.lsl', '.lslp'], 'LabVIEW': ['.lvproj'], 'Lasso': ['.lasso', '.las', '.lasso8', '.lasso9', '.ldml'], 'Latte': ['.latte'], 'Lean': ['.lean', '.hlean'], 'Less': ['.less'], 'Lex': ['.lex'], 'LilyPond': ['.ly', '.ily'], 'Linker Script': ['.ld', '.lds'], 'Liquid': ['.liquid'], 'Literate Agda': ['.lagda'], 'Literate CoffeeScript': ['.litcoffee'], 'Literate Haskell': ['.lhs'], 'LiveScript': ['.ls', '._ls'], 'Logos': ['.xm', '.x', '.xi'], 'Logtalk': ['.lgt', '.logtalk'], 'LookML': ['.lookml'], 'Lua': ['.lua', '.nse', '.pd_lua', '.rbxs', '.wlua'], 'M': ['.mumps'], 'M4': ['.m4'], 'MAXScript': ['.mcr'], 'MTML': ['.mtml'], 'MUF': ['.muf'], 'Makefile': ['.mak', '.mk', '.mkfile', 'Makefile'], 'Mako': ['.mako', '.mao'], 'Maple': ['.mpl'], 'Markdown': ['.md', '.markdown', '.mkd', '.mkdn', '.mkdown', '.ron'], 'Mask': ['.mask'], 'Mathematica': ['.mathematica', '.cdf', '.ma', '.mt', '.nb', '.nbp', '.wl', '.wlt'], 'Matlab': ['.matlab'], 'Max': ['.maxpat', '.maxhelp', '.maxproj', '.mxt', '.pat'], 'MediaWiki': ['.mediawiki', '.wiki'], 'Metal': ['.metal'], 'MiniD': ['.minid'], 'Mirah': ['.druby', '.duby', '.mir', '.mirah'], 'Modelica': ['.mo'], 'Module Management System': ['.mms', '.mmk'], 'Monkey': ['.monkey'], 'MoonScript': ['.moon'], 'Myghty': ['.myt'], 'NSIS': ['.nsi', '.nsh'], 'NetLinx': ['.axs', '.axi'], 'NetLinx+ERB': ['.axs.erb', '.axi.erb'], 'NetLogo': ['.nlogo'], 'Nginx': ['.nginxconf'], 'Nimrod': ['.nim', '.nimrod'], 'Ninja': ['.ninja'], 'Nit': ['.nit'], 'Nix': ['.nix'], 'Nu': ['.nu'], 'NumPy': ['.numpy', '.numpyw', '.numsc'], 'OCaml': ['.ml', '.eliom', '.eliomi', '.ml4', '.mli', '.mll', '.mly'], 'ObjDump': ['.objdump'], 'Objective-C++': ['.mm'], 'Objective-J': ['.sj'], 'Octave': ['.oct'], 'Omgrofl': ['.omgrofl'], 'Opa': ['.opa'], 'Opal': ['.opal'], 'OpenCL': ['.cl', '.opencl'], 'OpenEdge ABL': ['.p'], 'OpenSCAD': ['.scad'], 'Org': ['.org'], 'Ox': ['.ox', '.oxh', '.oxo'], 'Oxygene': ['.oxygene'], 'Oz': ['.oz'], 'PAWN': ['.pwn'], 'PHP': ['.php', '.aw', '.ctp', '.php3', '.php4', '.php5', '.phps', '.phpt'], 'POV-Ray SDL': ['.pov'], 'Pan': ['.pan'], 'Papyrus': ['.psc'], 'Parrot': ['.parrot'], 'Parrot Assembly': ['.pasm'], 'Parrot Internal Representation': ['.pir'], 'Pascal': ['.pas', '.dfm', '.dpr', '.lpr'], 'Perl': ['.pl', '.al', '.perl', '.ph', '.plx', '.pm', '.psgi', '.t'], 'Perl6': ['.6pl', '.6pm', '.nqp', '.p6', '.p6l', '.p6m', '.pl6', '.pm6'], 'Pickle': ['.pkl'], 'PigLatin': ['.pig'], 'Pike': ['.pike', '.pmod'], 'Pod': ['.pod'], 'PogoScript': ['.pogo'], 'Pony': ['.pony'], 'PostScript': ['.ps', '.eps'], 'PowerShell': ['.ps1', '.psd1', '.psm1'], 'Processing': ['.pde'], 'Prolog': ['.prolog', '.yap'], 'Propeller Spin': ['.spin'], 'Protocol Buffer': ['.proto'], 'Public Key': ['.pub'], 'Pure Data': ['.pd'], 'PureBasic': ['.pb', '.pbi'], 'PureScript': ['.purs'], 'Python': ['.py', '.bzl', '.gyp', '.lmi', '.pyde', '.pyp', '.pyt', '.pyw', '.tac', '.wsgi', '.xpy'], 'Python traceback': ['.pytb'], 'QML': ['.qml', '.qbs'], 'QMake': ['.pri'], 'R': ['.r', '.rd', '.rsx'], 'RAML': ['.raml'], 'RDoc': ['.rdoc'], 'REALbasic': ['.rbbas', '.rbfrm', '.rbmnu', '.rbres', '.rbtbar', '.rbuistate'], 'RHTML': ['.rhtml'], 'RMarkdown': ['.rmd'], 'Racket': ['.rkt', '.rktd', '.rktl', '.scrbl'], 'Ragel in Ruby Host': ['.rl'], 'Raw token data': ['.raw'], 'Rebol': ['.reb', '.r2', '.r3', '.rebol'], 'Red': ['.red', '.reds'], 'Redcode': ['.cw'], "Ren'Py": ['.rpy'], 'RenderScript': ['.rsh'], 'RobotFramework': ['.robot'], 'Rouge': ['.rg'], 'Ruby': ['.rb', '.builder', '.gemspec', '.god', '.irbrc', '.jbuilder', '.mspec', '.podspec', '.rabl', '.rake', '.rbuild', '.rbw', '.rbx', '.ru', '.ruby', '.thor', '.watchr'], 'Rust': ['.rs', '.rs.in'], 'SAS': ['.sas'], 'SCSS': ['.scss'], 'SMT': ['.smt2', '.smt'], 'SPARQL': ['.sparql', '.rq'], 'SQF': ['.sqf', '.hqf'], 'SQL': ['.pls', '.pck', '.pkb', '.pks', '.plb', '.plsql', '.sql', '.cql', '.ddl', '.prc', '.tab', '.udf', '.viw', '.db2'], 'STON': ['.ston'], 'SVG': ['.svg'], 'Sage': ['.sage', '.sagews'], 'SaltStack': ['.sls'], 'Sass': ['.sass'], 'Scala': ['.scala', '.sbt'], 'Scaml': ['.scaml'], 'Scheme': ['.scm', '.sld', '.sps', '.ss'], 'Scilab': ['.sci', '.sce'], 'Self': ['.self'], 'Shell': ['.sh', '.bash', '.bats', '.command', '.ksh', '.sh.in', '.tmux', '.tool', '.zsh'], 'ShellSession': ['.sh-session'], 'Shen': ['.shen'], 'Slash': ['.sl'], 'Slim': ['.slim'], 'Smali': ['.smali'], 'Smalltalk': ['.st'], 'Smarty': ['.tpl'], 'Solidity': ['.sol'], 'SourcePawn': ['.sp', '.sma'], 'Squirrel': ['.nut'], 'Stan': ['.stan'], 'Standard ML': ['.ML', '.fun', '.sig', '.sml'], 'Stata': ['.do', '.ado', '.doh', '.ihlp', '.mata', '.matah', '.sthlp'], 'Stylus': ['.styl'], 'SuperCollider': ['.scd'], 'Swift': ['.swift'], 'SystemVerilog': ['.sv', '.svh', '.vh'], 'TOML': ['.toml'], 'TXL': ['.txl'], 'Tcl': ['.tcl', '.adp', '.tm'], 'Tcsh': ['.tcsh', '.csh'], 'TeX': ['.tex', '.aux', '.bbx', '.bib', '.cbx', '.dtx', '.ins', '.lbx', '.ltx', '.mkii', '.mkiv', '.mkvi', '.sty', '.toc'], 'Tea': ['.tea'], 'Text': ['.txt', '.no'], 'Textile': ['.textile'], 'Thrift': ['.thrift'], 'Turing': ['.tu'], 'Turtle': ['.ttl'], 'Twig': ['.twig'], 'TypeScript': ['.ts', '.tsx'], 'Unified Parallel C': ['.upc'], 'Unity3D Asset': ['.anim', '.asset', '.mat', '.meta', '.prefab', '.unity'], 'Uno': ['.uno'], 'UnrealScript': ['.uc'], 'UrWeb': ['.ur', '.urs'], 'VCL': ['.vcl'], 'VHDL': ['.vhdl', '.vhd', '.vhf', '.vhi', '.vho', '.vhs', '.vht', '.vhw'], 'Vala': ['.vala', '.vapi'], 'Verilog': ['.veo'], 'VimL': ['.vim'], 'Visual Basic': ['.vb', '.bas', '.frm', '.frx', '.vba', '.vbhtml', '.vbs'], 'Volt': ['.volt'], 'Vue': ['.vue'], 'Web Ontology Language': ['.owl'], 'WebAssembly': ['.wat'], 'WebIDL': ['.webidl'], 'X10': ['.x10'], 'XC': ['.xc'], 'XML': ['.xml', '.ant', '.axml', '.ccxml', '.clixml', '.cproject', '.csl', '.csproj', '.ct', '.dita', '.ditamap', '.ditaval', '.dll.config', '.dotsettings', '.filters', '.fsproj', '.fxml', '.glade', '.grxml', '.iml', '.ivy', '.jelly', '.jsproj', '.kml', '.launch', '.mdpolicy', '.mxml', '.nproj', '.nuspec', '.odd', '.osm', '.plist', '.props', '.ps1xml', '.psc1', '.pt', '.rdf', '.rss', '.scxml', '.srdf', '.storyboard', '.stTheme', '.sublime-snippet', '.targets', '.tmCommand', '.tml', '.tmLanguage', '.tmPreferences', '.tmSnippet', '.tmTheme', '.ui', '.urdf', '.ux', '.vbproj', '.vcxproj', '.vssettings', '.vxml', '.wsdl', '.wsf', '.wxi', '.wxl', '.wxs', '.x3d', '.xacro', '.xaml', '.xib', '.xlf', '.xliff', '.xmi', '.xml.dist', '.xproj', '.xsd', '.xul', '.zcml'], 'XPages': ['.xsp-config', '.xsp.metadata'], 'XProc': ['.xpl', '.xproc'], 'XQuery': ['.xquery', '.xq', '.xql', '.xqm', '.xqy'], 'XS': ['.xs'], 'XSLT': ['.xslt', '.xsl'], 'Xojo': ['.xojo_code', '.xojo_menu', '.xojo_report', '.xojo_script', '.xojo_toolbar', '.xojo_window'], 'Xtend': ['.xtend'], 'YAML': ['.yml', '.reek', '.rviz', '.sublime-syntax', '.syntax', '.yaml', '.yaml-tmlanguage'], 'YANG': ['.yang'], 'Yacc': ['.y', '.yacc', '.yy'], 'Zephir': ['.zep'], 'Zig': ['.zig'], 'Zimpl': ['.zimpl', '.zmpl', '.zpl'], 'desktop': ['.desktop', '.desktop.in'], 'eC': ['.ec', '.eh'], 'edn': ['.edn'], 'fish': ['.fish'], 'mupad': ['.mu'], 'nesC': ['.nc'], 'ooc': ['.ooc'], 'reStructuredText': ['.rst', '.rest', '.rest.txt', '.rst.txt'], 'wisp': ['.wisp'], 'xBase': ['.prg', '.prw']}

DELETED_FILES_ = "Deleted files:\n"

MORE_MODIFIED_FILES_ = "Additional modified files (insufficient token budget to process):\n"

ADDED_FILES_ = "Additional added files (insufficient token budget to process):\n"

SYSTEM_PROMPT = """You are PR-Reviewer, a language model designed to review a Git Pull Request (PR).
Your task is to provide constructive and concise feedback for the PR.
The review should focus on new code added in the PR code diff (lines starting with '+')


The format we will use to present the PR code diff:
======
## File: 'src/file1.py'


@@ ... @@ def func1():
__new hunk__
11  unchanged code line0
12  unchanged code line1
13 +new code line2 added
14  unchanged code line3
__old hunk__
 unchanged code line0
 unchanged code line1
-old code line2 removed
 unchanged code line3

@@ ... @@ def func2():
__new hunk__
 unchanged code line4
+new code line5 removed
 unchanged code line6

## File: 'src/file2.py'
...
======

- In the format above, the diff is organized into separate '__new hunk__' and '__old hunk__' sections for each code chunk. '__new hunk__' contains the updated code, while '__old hunk__' shows the removed code. If no code was removed in a specific chunk, the __old hunk__ section will be omitted.
- We also added line numbers for the '__new hunk__' code, to help you refer to the code lines in your suggestions. These line numbers are not part of the actual code, and should only used for reference.
- Code lines are prefixed with symbols ('+', '-', ' '). The '+' symbol indicates new code added in the PR, the '-' symbol indicates code removed in the PR, and the ' ' symbol indicates unchanged code. The review should address new code added in the PR code diff (lines starting with '+')
- When quoting variables, names or file paths from the code, use backticks (`) instead of single quote (').


The output must be a YAML object equivalent to type $PRReview, according to the following Pydantic definitions:
=====

class KeyIssuesComponentLink(BaseModel):
    relevant_file: str = Field(description="The full file path of the relevant file")
    issue_header: str = Field(description="One or two word title for the issue. For example: 'Possible Bug', etc.")
    issue_content: str = Field(description="A short and concise summary of what should be further inspected and validated during the PR review process for this issue. Do not reference line numbers in this field.")
    start_line: int = Field(description="The start line that corresponds to this issue in the relevant file")
    end_line: int = Field(description="The end line that corresponds to this issue in the relevant file")

class Review(BaseModel):
    estimated_effort_to_review_[1-5]: int = Field(description="Estimate, on a scale of 1-5 (inclusive), the time and effort required to review this PR by an experienced and knowledgeable developer. 1 means short and easy review , 5 means long and hard review. Take into account the size, complexity, quality, and the needed changes of the PR code diff.")
    relevant_tests: str = Field(description="yes\\no question: does this PR have relevant tests added or updated ?")
    key_issues_to_review: List[KeyIssuesComponentLink] = Field("A short and diverse list (0-3 issues) of high-priority bugs, problems or performance concerns introduced in the PR code, which the PR reviewer should further focus on and validate during the review process.")
    security_concerns: str = Field(description="Does this PR code introduce possible vulnerabilities such as exposure of sensitive information (e.g., API keys, secrets, passwords), or security concerns like SQL injection, XSS, CSRF, and others ? Answer 'No' (without explaining why) if there are no possible issues. If there are security concerns or issues, start your answer with a short header, such as: 'Sensitive information exposure: ...', 'SQL injection: ...' etc. Explain your answer. Be specific and give examples if possible")

class PRReview(BaseModel):
    review: Review
=====


Example output:
```yaml
review:
  estimated_effort_to_review_[1-5]: |
    3
  relevant_tests: |
    No
  key_issues_to_review:
    - relevant_file: |
        directory/xxx.py
      issue_header: |
        Possible Bug
      issue_content: |
        ...
      start_line: 12
      end_line: 14
    - ...
  security_concerns: |
    No
```

Answer should be a valid YAML, and nothing else. Each YAML output MUST be after a newline, with proper indent, and block scalar indicator ('|')
"""

# SYSTEM_PROMPT = """You are PR-Reviewer, a language model designed to review a Git Pull Request (PR).
# Your task is to provide constructive and concise feedback for the PR.
# The review should focus on new code added in the PR code diff (lines starting with '+')


# The format we will use to present the PR code diff:
# ======
# ## File: 'src/file1.py'


# @@ ... @@ def func1():
# __new hunk__
# 11  unchanged code line0
# 12  unchanged code line1
# 13 +new code line2 added
# 14  unchanged code line3
# __old hunk__
#  unchanged code line0
#  unchanged code line1
# -old code line2 removed
#  unchanged code line3

# @@ ... @@ def func2():
# __new hunk__
#  unchanged code line4
# +new code line5 removed
#  unchanged code line6

# ## File: 'src/file2.py'
# ...
# ======

# - In the format above, the diff is organized into separate '__new hunk__' and '__old hunk__' sections for each code chunk. '__new hunk__' contains the updated code, while '__old hunk__' shows the removed code. If no code was removed in a specific chunk, the __old hunk__ section will be omitted.
# - We also added line numbers for the '__new hunk__' code, to help you refer to the code lines in your suggestions. These line numbers are not part of the actual code, and should only used for reference.
# - Code lines are prefixed with symbols ('+', '-', ' '). The '+' symbol indicates new code added in the PR, the '-' symbol indicates code removed in the PR, and the ' ' symbol indicates unchanged code. The review should address new code added in the PR code diff (lines starting with '+')
# - When quoting variables, names or file paths from the code, use backticks (`) instead of single quote (').


# The output must be in Markdown format, structured as follows:

# ## PR Review

# **Estimated Effort to Review (1-5):** [Estimate, on a scale of 1-5 (inclusive), the time and effort required to review this PR by an experienced and knowledgeable developer. 1 means short and easy review, 5 means long and hard review. Take into account the size, complexity, quality, and the needed changes of the PR code diff.]

# **Relevant Tests:** [Answer "Yes" or "No" whether this PR has relevant tests added or updated.]

# **Key Issues to Review:**
# *   **[Issue Header (e.g., Possible Bug)]** (`[relevant_file]`:[start_line]-[end_line]): [A short and concise summary of what should be further inspected and validated for this issue. Then print the relevant lines of code that are related to the issue.]
# *   ... (Provide a short and diverse list (0-3 issues) of high-priority bugs, problems, or performance concerns introduced in the PR code.)

# **Security Concerns:**
# [Describe any possible vulnerabilities such as exposure of sensitive information (e.g., API keys, secrets, passwords), or security concerns like SQL injection, XSS, CSRF, and others. Answer "No concerns found" if there are none. If there are concerns, start with a short header (e.g., **Sensitive information exposure:**) and explain, providing examples if possible.]


# Example output:
# ```markdown
# ## PR Review

# **Estimated Effort to Review (1-5):** 3

# **Relevant Tests:** No

# **Key Issues to Review:**
# *   **Possible Bug** (`directory/xxx.py`:12-14): The logic introduced might lead to an off-by-one error under certain conditions.
#     ```python
#     if (condition):
#         ...
#     ```
# *   **Performance** (`another/file.js`:88-92): This loop could be optimized by pre-calculating the value outside the loop.
#     ```python
#     for i in range(len(list)):
#         ...
#     ```
# **Security Concerns:**
# No concerns found.
# ```

# Ensure the output is valid Markdown and nothing else.
# """

# User Prompt - Provides the specific data for the review task
# USER_PROMPT = """
# --PR Info--

# Today's Date: {date}

# Title: '{title}'

# Branch: '{branch}'

# PR Description:
# ======
# {description}
# ======

# The PR code diff:
# ======
# {diff_content}
# ======


# Response (should be a valid YAML, and nothing else):
# ```yaml
# """
USER_PROMPT = """
--PR Info--

Title: '{title}'

Branch: '{branch}'


The PR code diff:
======
{diff_content}
======


Response (should be a valid YAML, and nothing else):
```yaml
"""

REFINE_PROMPT = """
You are PR-Review-Aggregator, a language model designed to synthesize multiple code review reports into a single, more accurate, and comprehensive report for a Git Pull Request (PR).

Your task is to analyze existing review reports for the same PR, along with the original PR code diff, and produce a consolidated review report. The goal is to leverage the insights from all reports to create a final report that is more robust, identifies the most critical issues, and resolves discrepancies.

You will be provided with:
1.  A list of review reports (in YAML format, following the $PRReview Pydantic definition).
2.  The original PR code diff.
3.  The PR title and branch information.

When generating the consolidated report, follow these guidelines for each field:

1.  **`estimated_effort_to_review_[1-5]`**:
    *   Consider the estimates from all reports.
    *   If they are similar, you can use the average or median.
    *   If they differ significantly, try to understand why (e.g., one report might have caught a complex issue others missed). Use your judgment to provide a reasoned estimate, potentially leaning towards a more cautious (higher) estimate if significant issues are raised by any report. Explain briefly if there's a major divergence and your reasoning.

2.  **`relevant_tests`**:
    *   If *any* of the reports indicate 'Yes', the consolidated answer should be 'Yes'.
    *   Only answer 'No' if *all* reports state 'No'.

3.  **`key_issues_to_review`**:
    *   This is the most critical part. Your goal is to produce a list of 0-3 *unique*, *high-priority* issues.
    *   **Aggregate:** Collect all `key_issues_to_review` from the reports.
    *   **De-duplicate & Consolidate:** Identify issues that refer to the same underlying problem, even if they are described differently or point to slightly different line numbers. Merge such issues into a single, more comprehensive entry. Ensure the `relevant_file`, `start_line`, and `end_line` are accurate for the consolidated issue, referring to the original PR code diff if needed.
    *   **Prioritize:** From the consolidated list, select the 0-3 most critical, impactful, or frequently mentioned issues. If multiple reports highlight the same or similar issues, it's a strong signal for inclusion.
    *   **Synthesize Content:** For each selected issue, synthesize the `issue_header` and `issue_content` from the input reports to be clear, concise, and actionable. If reports offer different perspectives on the same issue, try to incorporate the most valuable insights.

4.  **`security_concerns`**:
    *   If *any* of the reports identify a security concern, the consolidated report *must* address it.
    *   If multiple reports identify different security concerns, list them all or synthesize them into a comprehensive section.
    *   If multiple reports identify the *same* security concern, consolidate the explanation, providing the most complete and accurate description.
    *   If all reports state 'No' (or equivalent negative), then the consolidated answer can be 'No'. However, if even one report raises a credible concern, it should be included.

The format we will use to present the PR code diff (if you need to refer to it):
======
## File: 'src/file1.py'

@@ ... @@ def func1():
__new hunk__
11  unchanged code line0
12  unchanged code line1
13 +new code line2 added
14  unchanged code line3
__old hunk__
 unchanged code line0
 unchanged code line1
-old code line2 removed
 unchanged code line3
...
======
- In the format above, the diff is organized into separate '__new hunk__' and '__old hunk__' sections for each code chunk. '__new hunk__' contains the updated code, while '__old hunk__' shows the removed code.
- Line numbers are provided for the '__new hunk__' code for reference.
- Code lines are prefixed with symbols ('+', '-', ' '). The review should focus on new code added (lines starting with '+').
- When quoting variables, names or file paths from the code, use backticks (`) instead of single quote (').

The output must be a YAML object equivalent to type $PRReview, according to the following Pydantic definitions:
=====
class KeyIssuesComponentLink(BaseModel):
    relevant_file: str = Field(description="The full file path of the relevant file")
    issue_header: str = Field(description="One or two word title for the issue. For example: 'Possible Bug', etc.")
    issue_content: str = Field(description="A short and concise summary of what should be further inspected and validated during the PR review process for this issue. Do not reference line numbers in this field.")
    start_line: int = Field(description="The start line that corresponds to this issue in the relevant file")
    end_line: int = Field(description="The end line that corresponds to this issue in the relevant file")

class Review(BaseModel):
    estimated_effort_to_review_[1-5]: int = Field(description="Estimate, on a scale of 1-5 (inclusive), the time and effort required to review this PR by an experienced and knowledgeable developer. 1 means short and easy review , 5 means long and hard review. Take into account the size, complexity, quality, and the needed changes of the PR code diff, informed by the aggregated reports.")
    relevant_tests: str = Field(description="yes\\no question: does this PR have relevant tests added or updated, based on the input reports?")
    key_issues_to_review: List[KeyIssuesComponentLink] = Field("A short and diverse list (0-3 issues) of high-priority bugs, problems or performance concerns synthesized from the input reports, which the PR reviewer should further focus on and validate. Issues should be unique and consolidated.")
    security_concerns: str = Field(description="Does this PR code introduce possible vulnerabilities such as exposure of sensitive information, or security concerns like SQL injection, XSS, CSRF, etc., based on the input reports? Answer 'No' (without explaining why) if no reports indicate issues. If any report identifies security concerns, synthesize and detail them, starting with a short header.")

class PRReview(BaseModel):
    review: Review
=====

Example output:
```yaml
review:
  estimated_effort_to_review_[1-5]: |
    3
  relevant_tests: |
    No
  key_issues_to_review:
    - relevant_file: |
        directory/xxx.py
      issue_header: |
        Consolidated Bug
      issue_content: |
        This issue combines insights from report 1 and 3 regarding potential null pointer...
      start_line: 12
      end_line: 14
    - ...
  security_concerns: |
    No
```

Answer should be a valid YAML, and nothing else. Each YAML output MUST be after a newline, with proper indent, and block scalar indicator ('|').

--PR Info--
Title: '{title}'
Branch: '{branch}'

--Input Review Report List--
{review_report}

--Original PR Code Diff--
======
{diff_content}
======

Response (should be a valid YAML, and nothing else):
```yaml
"""

# --- Helper Functions ---

class EDIT_TYPE(Enum):
    ADDED = 1
    DELETED = 2
    MODIFIED = 3
    RENAMED = 4
    UNKNOWN = 5

@dataclass
class FilePatchInfo:
    base_file: str
    head_file: str
    patch: str
    filename: str
    tokens: int = -1
    edit_type: EDIT_TYPE = EDIT_TYPE.UNKNOWN
    old_filename: str = None
    num_plus_lines: int = -1
    num_minus_lines: int = -1
    language: Optional[str] = None
    ai_file_summary: str = None

class TokenHandler:
    def __init__(self, model: str, system_prompt: str, user_prompt: str):
        self.model = model
        self.prompt_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)

    def count_tokens(self, text: str) -> int:
        return int(len(text) / 3)


class TokenEncoder:
    _encoder_instance = None
    _model = None
    _lock = Lock()  # Create a lock object

    @classmethod
    def get_token_encoder(cls, model):
        if cls._encoder_instance is None or model != cls._model:  # Check without acquiring the lock for performance
            with cls._lock:  # Lock acquisition to ensure thread safety
                if cls._encoder_instance is None or model != cls._model:
                    cls._model = model
                    cls._encoder_instance = encoding_for_model(cls._model) if "gpt" in cls._model else get_encoding(
                        "cl100k_base")
        return cls._encoder_instance

def get_logger():
    return LOGGER

def set_max_tokens(max_tokens):
    global MAX_TOKENS
    MAX_TOKENS = max_tokens

def get_max_tokens(model):
    global MAX_TOKENS
    return MAX_TOKENS

def omit_deletion_hunks(patch_lines) -> str:
    """
    Omit deletion hunks from the patch and return the modified patch.
    Args:
    - patch_lines: a list of strings representing the lines of the patch
    Returns:
    - A string representing the modified patch with deletion hunks omitted
    """

    temp_hunk = []
    added_patched = []
    add_hunk = False
    inside_hunk = False
    RE_HUNK_HEADER = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?\ @@[ ]?(.*)")

    for line in patch_lines:
        if line.startswith('@@'):
            match = RE_HUNK_HEADER.match(line)
            if match:
                # finish previous hunk
                if inside_hunk and add_hunk:
                    added_patched.extend(temp_hunk)
                    temp_hunk = []
                    add_hunk = False
                temp_hunk.append(line)
                inside_hunk = True
        else:
            temp_hunk.append(line)
            if line:
                edit_type = line[0]
                if edit_type == '+':
                    add_hunk = True
    if inside_hunk and add_hunk:
        added_patched.extend(temp_hunk)

    return '\n'.join(added_patched)

def handle_patch_deletions(patch: str, original_file_content_str: str,
                           new_file_content_str: str, file_name: str, edit_type: EDIT_TYPE = EDIT_TYPE.UNKNOWN) -> str:
    """
    Handle entire file or deletion patches.

    This function takes a patch, original file content, new file content, and file name as input.
    It handles entire file or deletion patches and returns the modified patch with deletion hunks omitted.

    Args:
        patch (str): The patch to be handled.
        original_file_content_str (str): The original content of the file.
        new_file_content_str (str): The new content of the file.
        file_name (str): The name of the file.

    Returns:
        str: The modified patch with deletion hunks omitted.

    """
    if not new_file_content_str and (edit_type == EDIT_TYPE.DELETED or edit_type == EDIT_TYPE.UNKNOWN):
        # logic for handling deleted files - don't show patch, just show that the file was deleted
        patch = None # file was deleted
    else:
        patch_lines = patch.splitlines()
        patch_new = omit_deletion_hunks(patch_lines)
        if patch != patch_new:
            patch = patch_new
    return patch


def generate_full_patch(convert_hunks_to_line_numbers, file_dict, max_tokens_model,remaining_files_list_prev, token_handler):
    total_tokens = token_handler.prompt_tokens # initial tokens
    patches = []
    remaining_files_list_new = []
    files_in_patch_list = []
    for filename, data in file_dict.items():
        if filename not in remaining_files_list_prev:
            continue

        patch = data['patch']
        new_patch_tokens = data['tokens']
        edit_type = data['edit_type']

        # Hard Stop, no more tokens
        if total_tokens > max_tokens_model - OUTPUT_BUFFER_TOKENS_HARD_THRESHOLD:
            get_logger().warning(f"File was fully skipped, no more tokens: {filename}.")
            continue

        # If the patch is too large, just show the file name
        if total_tokens + new_patch_tokens > max_tokens_model - OUTPUT_BUFFER_TOKENS_SOFT_THRESHOLD:
            # Current logic is to skip the patch if it's too large
            # TODO: Option for alternative logic to remove hunks from the patch to reduce the number of tokens
            #  until we meet the requirements
            remaining_files_list_new.append(filename)
            continue

        if patch:
            if not convert_hunks_to_line_numbers:
                patch_final = f"\n\n## File: '{filename.strip()}'\n\n{patch.strip()}\n"
            else:
                patch_final = "\n\n" + patch.strip()
            patches.append(patch_final)
            total_tokens += token_handler.count_tokens(patch_final)
            files_in_patch_list.append(filename)
    return total_tokens, patches, remaining_files_list_new, files_in_patch_list

def pr_generate_compressed_diff(top_langs: list, token_handler: TokenHandler, model: str,
                                convert_hunks_to_line_numbers: bool,
                                large_pr_handling: bool) -> Tuple[list, list, list, list, dict, list]:
    deleted_files_list = []

    # sort each one of the languages in top_langs by the number of tokens in the diff
    sorted_files = []
    for lang in top_langs:
        sorted_files.extend(sorted(lang['files'], key=lambda x: x.tokens, reverse=True))

    # generate patches for each file, and count tokens
    file_dict = {}
    for file in sorted_files:
        original_file_content_str = file.base_file
        new_file_content_str = file.head_file
        patch = file.patch
        if not patch:
            continue

        # removing delete-only hunks
        patch = handle_patch_deletions(patch, original_file_content_str,
                                       new_file_content_str, file.filename, file.edit_type)
        if patch is None:
            if file.filename not in deleted_files_list:
                deleted_files_list.append(file.filename)
            continue

        if convert_hunks_to_line_numbers:
            patch = decouple_and_convert_to_hunks_with_lines_numbers(patch, file)

        ## add AI-summary metadata to the patch (disabled, since we are in the compressed diff)
        # if file.ai_file_summary and get_settings().config.get('config.is_auto_command', False):
        #     patch = add_ai_summary_top_patch(file, patch)

        new_patch_tokens = token_handler.count_tokens(patch)
        file_dict[file.filename] = {'patch': patch, 'tokens': new_patch_tokens, 'edit_type': file.edit_type}

    max_tokens_model = get_max_tokens(model)

    # first iteration
    files_in_patches_list = []
    remaining_files_list =  [file.filename for file in sorted_files]
    patches_list =[]
    total_tokens_list = []
    total_tokens, patches, remaining_files_list, files_in_patch_list = generate_full_patch(convert_hunks_to_line_numbers, file_dict,
                                       max_tokens_model, remaining_files_list, token_handler)
    patches_list.append(patches)
    total_tokens_list.append(total_tokens)
    files_in_patches_list.append(files_in_patch_list)

    # additional iterations (if needed)
    if large_pr_handling:
        for i in range(NUMBER_OF_ALLOWED_ITERATIONS-1):
            if remaining_files_list:
                total_tokens, patches, remaining_files_list, files_in_patch_list = generate_full_patch(convert_hunks_to_line_numbers,
                                                                                 file_dict,
                                                                                  max_tokens_model,
                                                                                  remaining_files_list, token_handler)
                if patches:
                    patches_list.append(patches)
                    total_tokens_list.append(total_tokens)
                    files_in_patches_list.append(files_in_patch_list)
            else:
                break

    return patches_list, total_tokens_list, deleted_files_list, remaining_files_list, file_dict, files_in_patches_list

def extract_hunk_headers(match):
    res = list(match.groups())
    for i in range(len(res)):
        if res[i] is None:
            res[i] = 0
    try:
        start1, size1, start2, size2 = map(int, res[:4])
    except:  # '@@ -0,0 +1 @@' case
        start1, size1, size2 = map(int, res[:3])
        start2 = 0
    section_header = res[4]
    return section_header, size1, size2, start1, start2

def decouple_and_convert_to_hunks_with_lines_numbers(patch: str, file) -> str:
    """
    Convert a given patch string into a string with line numbers for each hunk, indicating the new and old content of
    the file.

    Args:
        patch (str): The patch string to be converted.
        file: An object containing the filename of the file being patched.

    Returns:
        str: A string with line numbers for each hunk, indicating the new and old content of the file.

    example output:
    ## src/file.ts
    __new hunk__
    881        line1
    882        line2
    883        line3
    887 +      line4
    888 +      line5
    889        line6
    890        line7
    ...
    __old hunk__
            line1
            line2
    -       line3
    -       line4
            line5
            line6
            ...
    """

    # Add a header for the file
    if file:
        # if the file was deleted, return a message indicating that the file was deleted
        if hasattr(file, 'edit_type') and file.edit_type == EDIT_TYPE.DELETED:
            return f"\n\n## File '{file.filename.strip()}' was deleted\n"

        patch_with_lines_str = f"\n\n## File: '{file.filename.strip()}'\n"
    else:
        patch_with_lines_str = ""

    patch_lines = patch.splitlines()
    RE_HUNK_HEADER = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)")
    new_content_lines = []
    old_content_lines = []
    match = None
    start1, size1, start2, size2 = -1, -1, -1, -1
    prev_header_line = []
    header_line = []
    for line_i, line in enumerate(patch_lines):
        if 'no newline at end of file' in line.lower():
            continue

        if line.startswith('@@'):
            header_line = line
            match = RE_HUNK_HEADER.match(line)
            if match and (new_content_lines or old_content_lines):  # found a new hunk, split the previous lines
                if prev_header_line:
                    patch_with_lines_str += f'\n{prev_header_line}\n'
                is_plus_lines = is_minus_lines = False
                if new_content_lines:
                    is_plus_lines = any([line.startswith('+') for line in new_content_lines])
                if old_content_lines:
                    is_minus_lines = any([line.startswith('-') for line in old_content_lines])
                if is_plus_lines or is_minus_lines: # notice 'True' here - we always present __new hunk__ for section, otherwise LLM gets confused
                    patch_with_lines_str = patch_with_lines_str.rstrip() + '\n__new hunk__\n'
                    for i, line_new in enumerate(new_content_lines):
                        patch_with_lines_str += f"{start2 + i} {line_new}\n"
                if is_minus_lines:
                    patch_with_lines_str = patch_with_lines_str.rstrip() + '\n__old hunk__\n'
                    for line_old in old_content_lines:
                        patch_with_lines_str += f"{line_old}\n"
                new_content_lines = []
                old_content_lines = []
            if match:
                prev_header_line = header_line

            section_header, size1, size2, start1, start2 = extract_hunk_headers(match)

        elif line.startswith('+'):
            new_content_lines.append(line)
        elif line.startswith('-'):
            old_content_lines.append(line)
        else:
            if not line and line_i: # if this line is empty and the next line is a hunk header, skip it
                if line_i + 1 < len(patch_lines) and patch_lines[line_i + 1].startswith('@@'):
                    continue
                elif line_i + 1 == len(patch_lines):
                    continue
            new_content_lines.append(line)
            old_content_lines.append(line)

    # finishing last hunk
    if match and new_content_lines:
        patch_with_lines_str += f'\n{header_line}\n'
        is_plus_lines = is_minus_lines = False
        if new_content_lines:
            is_plus_lines = any([line.startswith('+') for line in new_content_lines])
        if old_content_lines:
            is_minus_lines = any([line.startswith('-') for line in old_content_lines])
        if is_plus_lines or is_minus_lines:  # notice 'True' here - we always present __new hunk__ for section, otherwise LLM gets confused
            patch_with_lines_str = patch_with_lines_str.rstrip() + '\n__new hunk__\n'
            for i, line_new in enumerate(new_content_lines):
                patch_with_lines_str += f"{start2 + i} {line_new}\n"
        if is_minus_lines:
            patch_with_lines_str = patch_with_lines_str.rstrip() + '\n__old hunk__\n'
            for line_old in old_content_lines:
                patch_with_lines_str += f"{line_old}\n"

    return patch_with_lines_str.rstrip()

def should_skip_patch(filename):
    patch_extension_skip_types = ['.md', '.txt', '.png', '.pdf']
    if patch_extension_skip_types and filename:
        return any(filename.endswith(skip_type) for skip_type in patch_extension_skip_types)
    return False

def check_if_hunk_lines_matches_to_file(i, original_lines, patch_lines, start1):
    """
    Check if the hunk lines match the original file content. We saw cases where the hunk header line doesn't match the original file content, and then
    extending the hunk with extra lines before the hunk header can cause the hunk to be invalid.
    """
    is_valid_hunk = True
    try:
        if i + 1 < len(patch_lines) and patch_lines[i + 1][0] == ' ': # an existing line in the file
            if patch_lines[i + 1].strip() != original_lines[start1 - 1].strip():
                # check if different encoding is needed
                original_line = original_lines[start1 - 1].strip()
                for encoding in ['iso-8859-1', 'latin-1', 'ascii', 'utf-16']:
                    try:
                        if original_line.encode(encoding).decode().strip() == patch_lines[i + 1].strip():
                            get_logger().info(f"Detected different encoding in hunk header line {start1}, needed encoding: {encoding}")
                            return False # we still want to avoid extending the hunk. But we don't want to log an error
                    except:
                        pass

                is_valid_hunk = False
                get_logger().info(
                    f"Invalid hunk in PR, line {start1} in hunk header doesn't match the original file content")
    except:
        pass
    return is_valid_hunk

def process_patch_lines(patch_str, original_file_str, patch_extra_lines_before, patch_extra_lines_after, new_file_str=""):
    allow_dynamic_context = True
    patch_extra_lines_before_dynamic = 8

    file_original_lines = original_file_str.splitlines()
    file_new_lines = new_file_str.splitlines() if new_file_str else []
    len_original_lines = len(file_original_lines)
    patch_lines = patch_str.splitlines()
    extended_patch_lines = []

    is_valid_hunk = True
    start1, size1, start2, size2 = -1, -1, -1, -1
    RE_HUNK_HEADER = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)")
    try:
        for i,line in enumerate(patch_lines):
            if line.startswith('@@'):
                match = RE_HUNK_HEADER.match(line)
                # identify hunk header
                if match:
                    # finish processing previous hunk
                    if is_valid_hunk and (start1 != -1 and patch_extra_lines_after > 0):
                        delta_lines_original = [f' {line}' for line in file_original_lines[start1 + size1 - 1:start1 + size1 - 1 + patch_extra_lines_after]]
                        extended_patch_lines.extend(delta_lines_original)

                    section_header, size1, size2, start1, start2 = extract_hunk_headers(match)

                    is_valid_hunk = check_if_hunk_lines_matches_to_file(i, file_original_lines, patch_lines, start1)

                    if is_valid_hunk and (patch_extra_lines_before > 0 or patch_extra_lines_after > 0):
                        def _calc_context_limits(patch_lines_before):
                            extended_start1 = max(1, start1 - patch_lines_before)
                            extended_size1 = size1 + (start1 - extended_start1) + patch_extra_lines_after
                            extended_start2 = max(1, start2 - patch_lines_before)
                            extended_size2 = size2 + (start2 - extended_start2) + patch_extra_lines_after
                            if extended_start1 - 1 + extended_size1 > len_original_lines:
                                # we cannot extend beyond the original file
                                delta_cap = extended_start1 - 1 + extended_size1 - len_original_lines
                                extended_size1 = max(extended_size1 - delta_cap, size1)
                                extended_size2 = max(extended_size2 - delta_cap, size2)
                            return extended_start1, extended_size1, extended_start2, extended_size2

                        if allow_dynamic_context and file_new_lines:
                            extended_start1, extended_size1, extended_start2, extended_size2 = \
                                _calc_context_limits(patch_extra_lines_before_dynamic)

                            lines_before_original = file_original_lines[extended_start1 - 1:start1 - 1]
                            lines_before_new = file_new_lines[extended_start2 - 1:start2 - 1]
                            found_header = False
                            if lines_before_original == lines_before_new: # Making sure no changes from a previous hunk
                                for i, line, in enumerate(lines_before_original):
                                    if section_header in line:
                                        found_header = True
                                        # Update start and size in one line each
                                        extended_start1, extended_start2 = extended_start1 + i, extended_start2 + i
                                        extended_size1, extended_size2 = extended_size1 - i, extended_size2 - i
                                        # get_logger().debug(f"Found section header in line {i} before the hunk")
                                        section_header = ''
                                        break
                            else:
                                get_logger().debug(f"Extra lines before hunk are different in original and new file - dynamic context",
                                                   artifact={"lines_before_original": lines_before_original,
                                                             "lines_before_new": lines_before_new})

                            if not found_header:
                                # get_logger().debug(f"Section header not found in the extra lines before the hunk")
                                extended_start1, extended_size1, extended_start2, extended_size2 = \
                                    _calc_context_limits(patch_extra_lines_before)
                        else:
                            extended_start1, extended_size1, extended_start2, extended_size2 = \
                                _calc_context_limits(patch_extra_lines_before)

                        # check if extra lines before hunk are different in original and new file
                        delta_lines_original = [f' {line}' for line in file_original_lines[extended_start1 - 1:start1 - 1]]
                        if file_new_lines:
                            delta_lines_new = [f' {line}' for line in file_new_lines[extended_start2 - 1:start2 - 1]]
                            if delta_lines_original != delta_lines_new:
                                get_logger().debug(f"Extra lines before hunk are different in original and new file",
                                                   artifact={"delta_lines_original": delta_lines_original,
                                                             "delta_lines_new": delta_lines_new})
                                extended_start1 = start1
                                extended_size1 = size1
                                extended_start2 = start2
                                extended_size2 = size2
                                delta_lines_original = []

                        #  logic to remove section header if its in the extra delta lines (in dynamic context, this is also done)
                        if section_header and not allow_dynamic_context:
                            for line in delta_lines_original:
                                if section_header in line:
                                    section_header = ''  # remove section header if it is in the extra delta lines
                                    break
                    else:
                        extended_start1 = start1
                        extended_size1 = size1
                        extended_start2 = start2
                        extended_size2 = size2
                        delta_lines_original = []
                    extended_patch_lines.append('')
                    extended_patch_lines.append(
                        f'@@ -{extended_start1},{extended_size1} '
                        f'+{extended_start2},{extended_size2} @@ {section_header}')
                    extended_patch_lines.extend(delta_lines_original)  # one to zero based
                    continue
            extended_patch_lines.append(line)
    except Exception as e:
        get_logger().warning(f"Failed to extend patch: {e}", artifact={"traceback": traceback.format_exc()})
        return patch_str

    # finish processing last hunk
    if start1 != -1 and patch_extra_lines_after > 0 and is_valid_hunk:
        delta_lines_original = file_original_lines[start1 + size1 - 1:start1 + size1 - 1 + patch_extra_lines_after]
        # add space at the beginning of each extra line
        delta_lines_original = [f' {line}' for line in delta_lines_original]
        extended_patch_lines.extend(delta_lines_original)

    extended_patch_str = '\n'.join(extended_patch_lines)
    return extended_patch_str

def extend_patch(original_file_str, patch_str, patch_extra_lines_before=0,
                 patch_extra_lines_after=0, filename: str = "", new_file_str="") -> str:
    if not patch_str or (patch_extra_lines_before == 0 and patch_extra_lines_after == 0) or not original_file_str:
        return patch_str

    original_file_str = decode_if_bytes(original_file_str)
    new_file_str = decode_if_bytes(new_file_str)
    if not original_file_str:
        return patch_str

    if should_skip_patch(filename):
        return patch_str

    try:
        extended_patch_str = process_patch_lines(patch_str, original_file_str,
                                                 patch_extra_lines_before, patch_extra_lines_after, new_file_str)
    except Exception as e:
        get_logger().warning(f"Failed to extend patch: {e}", artifact={"traceback": traceback.format_exc()})
        return patch_str

    return extended_patch_str


def decode_if_bytes(original_file_str):
    if isinstance(original_file_str, (bytes, bytearray)):
        try:
            return original_file_str.decode('utf-8')
        except UnicodeDecodeError:
            encodings_to_try = ['iso-8859-1', 'latin-1', 'ascii', 'utf-16']
            for encoding in encodings_to_try:
                try:
                    return original_file_str.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return ""
    return original_file_str

def pr_generate_extended_diff(pr_languages: list,
                              token_handler: TokenHandler,
                              add_line_numbers_to_hunks: bool,
                              patch_extra_lines_before: int = 0,
                              patch_extra_lines_after: int = 0) -> Tuple[list, int, list]:
    total_tokens = token_handler.prompt_tokens  # initial tokens
    patches_extended = []
    patches_extended_tokens = []
    for lang in pr_languages:
        for file in lang['files']:
            original_file_content_str = file.base_file
            new_file_content_str = file.head_file
            patch = file.patch
            if not patch:
                continue

            # extend each patch with extra lines of context
            extended_patch = extend_patch(original_file_content_str, patch,
                                          patch_extra_lines_before, patch_extra_lines_after, file.filename,
                                          new_file_str=new_file_content_str)
            if not extended_patch:
                get_logger().warning(f"Failed to extend patch for file: {file.filename}")
                continue

            if add_line_numbers_to_hunks:
                full_extended_patch = decouple_and_convert_to_hunks_with_lines_numbers(extended_patch, file)
            else:
                extended_patch = extended_patch.replace('\n@@ ', '\n\n@@ ') # add extra line before each hunk
                full_extended_patch = f"\n\n## File: '{file.filename.strip()}'\n\n{extended_patch.strip()}\n"

            patch_tokens = token_handler.count_tokens(full_extended_patch)
            file.tokens = patch_tokens
            total_tokens += patch_tokens
            patches_extended_tokens.append(patch_tokens)
            patches_extended.append(full_extended_patch)

    return patches_extended, total_tokens, patches_extended_tokens


def filter_bad_extensions(files):
    return [f for f in files if f.filename is not None and is_valid_file(f.filename, BAD_EXTENSIONS)]


def is_valid_file(filename:str, bad_extensions=None) -> bool:
    if not filename:
        return False

    auto_generated_files = ['package-lock.json', 'yarn.lock', 'composer.lock', 'Gemfile.lock', 'poetry.lock']
    for forbidden_file in auto_generated_files:
        if filename.endswith(forbidden_file):
            return False

    return filename.split('.')[-1] not in bad_extensions

def sort_files_by_main_languages(languages: Dict, files: list):
    """
    Sort files by their main language, put the files that are in the main language first and the rest files after
    """
    # sort languages by their size
    languages_sorted_list = [k for k, v in sorted(languages.items(), key=lambda item: item[1], reverse=True)]
    # languages_sorted = sorted(languages, key=lambda x: x[1], reverse=True)
    # get all extensions for the languages
    main_extensions = []
    language_extension_map_org = LANGUAGE_EXTENSION_MAP
    language_extension_map = {k.lower(): v for k, v in language_extension_map_org.items()}
    for language in languages_sorted_list:
        if language.lower() in language_extension_map:
            main_extensions.append(language_extension_map[language.lower()])
        else:
            main_extensions.append([])

    # filter out files bad extensions
    files_filtered = filter_bad_extensions(files)

    # sort files by their extension, put the files that are in the main extension first
    # and the rest files after, map languages_sorted to their respective files
    files_sorted = []
    rest_files = {}

    # if no languages detected, put all files in the "Other" category
    if not languages:
        files_sorted = [({"language": "Other", "files": list(files_filtered)})]
        return files_sorted

    main_extensions_flat = []
    for ext in main_extensions:
        main_extensions_flat.extend(ext)

    for extensions, lang in zip(main_extensions, languages_sorted_list):  # noqa: B905
        tmp = []
        for file in files_filtered:
            extension_str = f".{file.filename.split('.')[-1]}"
            if extension_str in extensions:
                tmp.append(file)
            else:
                if (file.filename not in rest_files) and (extension_str not in main_extensions_flat):
                    rest_files[file.filename] = file
        if len(tmp) > 0:
            files_sorted.append({"language": lang, "files": tmp})
    files_sorted.append({"language": "Other", "files": list(rest_files.values())})
    return files_sorted

def get_diff_files(repo: git.Repo, head_branch_name: str, target_branch_name: str) -> list[FilePatchInfo]:
    diffs = repo.branches[target_branch_name].commit.diff(
        repo.branches[head_branch_name].commit,
        create_patch=True,
        R=True
    )
    diff_files = []
    for diff_item in diffs:
        try:
            if diff_item.a_blob is not None:
                original_file_content_str = diff_item.a_blob.data_stream.read().decode('utf-8')
            else:
                original_file_content_str = ""  # empty file
            if diff_item.b_blob is not None:
                new_file_content_str = diff_item.b_blob.data_stream.read().decode('utf-8')
            else:
                new_file_content_str = ""  # empty file
            edit_type = EDIT_TYPE.MODIFIED
            if diff_item.new_file:
                edit_type = EDIT_TYPE.ADDED
            elif diff_item.deleted_file:
                edit_type = EDIT_TYPE.DELETED
            elif diff_item.renamed_file:
                edit_type = EDIT_TYPE.RENAMED
            diff_files.append(
                FilePatchInfo(original_file_content_str,
                            new_file_content_str,
                            diff_item.diff.decode('utf-8'),
                            diff_item.b_path,
                            edit_type=edit_type,
                            old_filename=None if diff_item.a_path == diff_item.b_path else diff_item.a_path
                            )
            )
        except Exception as e:
            diff_path = diff_item.b_path if diff_item.b_path else diff_item.a_path
            get_logger().warning(f"Skipping file {diff_path} due to error: {e.__class__.__name__}: {e}")
    return diff_files

def get_languages(repo: git.Repo) -> dict:
    """
    Calculate percentage of languages in repository. Used for hunk prioritisation.
    """
    # Get all files in repository
    filepaths = [Path(item.path) for item in repo.tree().traverse() if item.type == 'blob']
    # Identify language by file extension and count
    lang_count = Counter(ext.lstrip('.') for filepath in filepaths for ext in [filepath.suffix.lower()])
    # Convert counts to percentages
    total_files = len(filepaths)
    lang_percentage = {lang: count / total_files * 100 for lang, count in lang_count.items()}
    return lang_percentage

def clip_tokens(model: str, text: str, max_tokens: int, add_three_dots=True, num_input_tokens=None, delete_last_line=False) -> str:
    """
    Clip the number of tokens in a string to a maximum number of tokens.

    Args:
        text (str): The string to clip.
        max_tokens (int): The maximum number of tokens allowed in the string.
        add_three_dots (bool, optional): A boolean indicating whether to add three dots at the end of the clipped
    Returns:
        str: The clipped string.
    """
    if not text:
        return text

    try:
        if num_input_tokens is None:
            encoder = TokenEncoder.get_token_encoder(model)
            num_input_tokens = len(encoder.encode(text))
        if num_input_tokens <= max_tokens:
            return text
        if max_tokens < 0:
            return ""

        # calculate the number of characters to keep
        num_chars = len(text)
        chars_per_token = num_chars / num_input_tokens
        factor = 0.9  # reduce by 10% to be safe
        num_output_chars = int(factor * chars_per_token * max_tokens)

        # clip the text
        if num_output_chars > 0:
            clipped_text = text[:num_output_chars]
            if delete_last_line:
                clipped_text = clipped_text.rsplit('\n', 1)[0]
            if add_three_dots:
                clipped_text += "\n...(truncated)"
        else: # if the text is empty
            clipped_text =  ""

        return clipped_text
    except Exception as e:
        get_logger().warning(f"Failed to clip tokens: {e}")
        return text

def get_pr_diff_old(repo_path: str, base_branch_name: str, target_branch_name: str, 
                token_handler: TokenHandler, model: str,
                add_line_numbers_to_hunks: bool = False,
                large_pr_handling: bool = False):
    PATCH_EXTRA_LINES_BEFORE = 3
    PATCH_EXTRA_LINES_AFTER = 1

    repo = git.Repo(repo_path)
    if repo.is_dirty():
        raise ValueError('The repository is not in a clean state. Please commit or stash pending changes.')
    if target_branch_name not in repo.heads:
        raise KeyError(f'Branch: {target_branch_name} does not exist')
    
    diff_files = get_diff_files(repo, base_branch_name, target_branch_name)

    # get pr languages
    pr_languages = sort_files_by_main_languages(get_languages(repo), diff_files)
    if pr_languages:
        try:
            get_logger().info(f"PR main language: {pr_languages[0]['language']}")
        except Exception as e:
            pass

    # generate a standard diff string, with patch extension
    patches_extended, total_tokens, patches_extended_tokens = pr_generate_extended_diff(
        pr_languages, token_handler, add_line_numbers_to_hunks,
        patch_extra_lines_before=PATCH_EXTRA_LINES_BEFORE, patch_extra_lines_after=PATCH_EXTRA_LINES_AFTER)
    
    patches_extended = "\n".join(patches_extended)
    # if we are under the limit, return the full diff
    if total_tokens + OUTPUT_BUFFER_TOKENS_SOFT_THRESHOLD < get_max_tokens(model):
        get_logger().info(f"Tokens: {total_tokens}, total tokens under limit: {get_max_tokens(model)}, returning full diff.")
        return patches_extended, diff_files
    # else:
    #     patches_extended = clip_tokens(model, patches_extended, get_max_tokens(model))
    #     return patches_extended, diff_files
    
    # if we are over the limit, start pruning (If we got here, we will not extend the patches with extra lines)
    get_logger().info(f"Tokens: {total_tokens}, total tokens over limit: {get_max_tokens(model)}, pruning diff.")
    patches_compressed_list, total_tokens_list, deleted_files_list, remaining_files_list, file_dict, files_in_patches_list = \
        pr_generate_compressed_diff(pr_languages, token_handler, model, add_line_numbers_to_hunks, large_pr_handling)

    if large_pr_handling and len(patches_compressed_list) > 1:
        get_logger().info(f"Large PR handling mode, and found {len(patches_compressed_list)} patches with original diff.")
        return "", diff_files # return empty string, as we want to generate multiple patches with a different prompt

    # return the first patch
    patches_compressed = patches_compressed_list[0]
    total_tokens_new = total_tokens_list[0]
    files_in_patch = files_in_patches_list[0]

    # Insert additional information about added, modified, and deleted files if there is enough space
    max_tokens = get_max_tokens(model) - OUTPUT_BUFFER_TOKENS_HARD_THRESHOLD
    curr_token = total_tokens_new  # == token_handler.count_tokens(final_diff)+token_handler.prompt_tokens
    final_diff = "\n".join(patches_compressed)
    delta_tokens = 10
    added_list_str = modified_list_str = deleted_list_str = ""
    unprocessed_files = []
    # generate the added, modified, and deleted files lists
    if (max_tokens - curr_token) > delta_tokens:
        for filename, file_values in file_dict.items():
            if filename in files_in_patch:
                continue
            if file_values['edit_type'] == EDIT_TYPE.ADDED:
                unprocessed_files.append(filename)
                if not added_list_str:
                    added_list_str = ADDED_FILES_ + f"\n{filename}"
                else:
                    added_list_str = added_list_str + f"\n{filename}"
            elif file_values['edit_type'] in [EDIT_TYPE.MODIFIED, EDIT_TYPE.RENAMED]:
                unprocessed_files.append(filename)
                if not modified_list_str:
                    modified_list_str = MORE_MODIFIED_FILES_ + f"\n{filename}"
                else:
                    modified_list_str = modified_list_str + f"\n{filename}"
            elif file_values['edit_type'] == EDIT_TYPE.DELETED:
                # unprocessed_files.append(filename) # not needed here, because the file was deleted, so no need to process it
                if not deleted_list_str:
                    deleted_list_str = DELETED_FILES_ + f"\n{filename}"
                else:
                    deleted_list_str = deleted_list_str + f"\n{filename}"

    # prune the added, modified, and deleted files lists, and add them to the final diff
    added_list_str = clip_tokens(model, added_list_str, max_tokens - curr_token)
    if added_list_str:
        final_diff = final_diff + "\n\n" + added_list_str
        curr_token += token_handler.count_tokens(added_list_str) + 2
    modified_list_str = clip_tokens(model, modified_list_str, max_tokens - curr_token)
    if modified_list_str:
        final_diff = final_diff + "\n\n" + modified_list_str
        curr_token += token_handler.count_tokens(modified_list_str) + 2
    deleted_list_str = clip_tokens(model, deleted_list_str, max_tokens - curr_token)
    if deleted_list_str:
        final_diff = final_diff + "\n\n" + deleted_list_str

    get_logger().debug(f"After pruning, added_list_str: {added_list_str}, modified_list_str: {modified_list_str}, "
                       f"deleted_list_str: {deleted_list_str}")
    return final_diff, diff_files

def get_pr_diff(item):
    pr_code_changes = ""
    # import pdb; pdb.set_trace()
    for commit in item["pr_commits"]:
        commit_code_change = f"Commit: {commit['sha']}\n"
        commit_code_change += f"Commit Message: {commit['message']}\n"
        commit_code_change += f"Commit Code Changes: \n"
        for diff in commit["diff"]:
            commit_code_change += f"{diff['file']}\n"
            commit_code_change += f"```\n{diff['patch']}\n```\n"
        pr_code_changes += f"{commit_code_change}\n"
    pr_code_changes = pr_code_changes.strip()
    return pr_code_changes

def load_yaml(response_text: str, keys_fix_yaml: List[str] = [], first_key="", last_key="") -> dict:
    response_text_original = copy.deepcopy(response_text)
    response_text = response_text.strip('\n').removeprefix('```yaml').rstrip().removesuffix('```')
    try:
        data = yaml.safe_load(response_text)
    except Exception as e:
        get_logger().warning(f"Initial failure to parse AI prediction: {e}")
        data = try_fix_yaml(response_text, keys_fix_yaml=keys_fix_yaml, first_key=first_key, last_key=last_key,
                            response_text_original=response_text_original)
        if not data:
            get_logger().error(f"Failed to parse AI prediction after fallbacks",
                               artifact={'response_text': response_text})
        else:
            get_logger().info(f"Successfully parsed AI prediction after fallbacks",
                              artifact={'response_text': response_text})
    return data



def try_fix_yaml(response_text: str,
                 keys_fix_yaml: List[str] = [],
                 first_key="",
                 last_key="",
                 response_text_original="") -> dict:
    response_text_lines = response_text.split('\n')

    keys_yaml = ['relevant line:', 'suggestion content:', 'relevant file:', 'existing code:', 'improved code:']
    keys_yaml = keys_yaml + keys_fix_yaml
    # first fallback - try to convert 'relevant line: ...' to relevant line: |-\n        ...'
    response_text_lines_copy = response_text_lines.copy()
    for i in range(0, len(response_text_lines_copy)):
        for key in keys_yaml:
            if key in response_text_lines_copy[i] and not '|' in response_text_lines_copy[i]:
                response_text_lines_copy[i] = response_text_lines_copy[i].replace(f'{key}',
                                                                                  f'{key} |\n        ')
    try:
        data = yaml.safe_load('\n'.join(response_text_lines_copy))
        get_logger().info(f"Successfully parsed AI prediction after adding |-\n")
        return data
    except:
        pass

    # second fallback - try to extract only range from first ```yaml to ````
    snippet_pattern = r'```(yaml)?[\s\S]*?```'
    snippet = re.search(snippet_pattern, '\n'.join(response_text_lines_copy))
    if not snippet:
        snippet = re.search(snippet_pattern, response_text_original) # before we removed the "```"
    if snippet:
        snippet_text = snippet.group()
        try:
            data = yaml.safe_load(snippet_text.removeprefix('```yaml').rstrip('`'))
            get_logger().info(f"Successfully parsed AI prediction after extracting yaml snippet")
            return data
        except:
            pass


    # third fallback - try to remove leading and trailing curly brackets
    response_text_copy = response_text.strip().rstrip().removeprefix('{').removesuffix('}').rstrip(':\n')
    try:
        data = yaml.safe_load(response_text_copy)
        get_logger().info(f"Successfully parsed AI prediction after removing curly brackets")
        return data
    except:
        pass


    # forth fallback - try to extract yaml snippet by 'first_key' and 'last_key'
    # note that 'last_key' can be in practice a key that is not the last key in the yaml snippet.
    # it just needs to be some inner key, so we can look for newlines after it
    if first_key and last_key:
        index_start = response_text.find(f"\n{first_key}:")
        if index_start == -1:
            index_start = response_text.find(f"{first_key}:")
        index_last_code = response_text.rfind(f"{last_key}:")
        index_end = response_text.find("\n\n", index_last_code) # look for newlines after last_key
        if index_end == -1:
            index_end = len(response_text)
        response_text_copy = response_text[index_start:index_end].strip().strip('```yaml').strip('`').strip()
        try:
            data = yaml.safe_load(response_text_copy)
            get_logger().info(f"Successfully parsed AI prediction after extracting yaml snippet")
            return data
        except:
            pass

    # fifth fallback - try to remove leading '+' (sometimes added by AI for 'existing code' and 'improved code')
    response_text_lines_copy = response_text_lines.copy()
    for i in range(0, len(response_text_lines_copy)):
        if response_text_lines_copy[i].startswith('+'):
            response_text_lines_copy[i] = ' ' + response_text_lines_copy[i][1:]
    try:
        data = yaml.safe_load('\n'.join(response_text_lines_copy))
        get_logger().info(f"Successfully parsed AI prediction after removing leading '+'")
        return data
    except:
        pass

    # sixth fallback - try to remove last lines
    for i in range(1, len(response_text_lines)):
        response_text_lines_tmp = '\n'.join(response_text_lines[:-i])
        try:
            data = yaml.safe_load(response_text_lines_tmp)
            get_logger().info(f"Successfully parsed AI prediction after removing {i} lines")
            return data
        except:
            pass

def is_value_no(value):
    if not value:
        return True
    value_str = str(value).strip().lower()
    if value_str == 'no' or value_str == 'none' or value_str == 'false':
        return True
    return False

def emphasize_header(text: str, only_markdown=False, reference_link=None) -> str:
    try:
        # Finding the position of the first occurrence of ": "
        colon_position = text.find(": ")

        # Splitting the string and wrapping the first part in <strong> tags
        if colon_position != -1:
            # Everything before the colon (inclusive) is wrapped in <strong> tags
            if only_markdown:
                if reference_link:
                    transformed_string = f"[**{text[:colon_position + 1]}**]({reference_link})\n" + text[colon_position + 1:]
                else:
                    transformed_string = f"**{text[:colon_position + 1]}**\n" + text[colon_position + 1:]
            else:
                if reference_link:
                    transformed_string = f"<strong><a href='{reference_link}'>{text[:colon_position + 1]}</a></strong><br>" + text[colon_position + 1:]
                else:
                    transformed_string = "<strong>" + text[:colon_position + 1] + "</strong>" +'<br>' + text[colon_position + 1:]
        else:
            # If there's no ": ", return the original string
            transformed_string = text

        return transformed_string
    except Exception as e:
        get_logger().exception(f"Failed to emphasize header: {e}")
        return text

def convert_to_markdown_v2(output_data: dict,
                           gfm_supported: bool = True,
                           files=None) -> str:
    """
    Convert a dictionary of data into markdown format.
    Args:
        output_data (dict): A dictionary containing data to be converted to markdown format.
    Returns:
        str: The markdown formatted text generated from the input dictionary.
    """

    emojis = {
        "Can be split": "🔀",
        "Key issues to review": "⚡",
        "Recommended focus areas for review": "⚡",
        "Score": "🏅",
        "Relevant tests": "🧪",
        "Focused PR": "✨",
        "Relevant ticket": "🎫",
        "Security concerns": "🔒",
        "Insights from user's answers": "📝",
        "Code feedback": "🤖",
        "Estimated effort to review [1-5]": "⏱️",
        "Ticket compliance check": "🎫",
    }
    markdown_text = ""
    markdown_text += "## PR Reviewer Guide 🔍\n\n"

    if not output_data or not output_data.get('review', {}):
        return ""

    markdown_text += f"Here are some key observations to aid the review process:\n\n"

    if gfm_supported:
        markdown_text += "<table>\n"

    for key, value in output_data['review'].items():
        if value is None or value == '' or value == {} or value == []:
            if key.lower() not in ['can_be_split', 'key_issues_to_review']:
                continue
        key_nice = key.replace('_', ' ').capitalize()
        emoji = emojis.get(key_nice, "")
        if 'Estimated effort to review' in key_nice:
            key_nice = 'Estimated effort to review'
            value = str(value).strip()
            if value.isnumeric():
                value_int = int(value)
            else:
                try:
                    value_int = int(value.split(',')[0])
                except ValueError:
                    continue
            blue_bars = '🔵' * value_int
            white_bars = '⚪' * (5 - value_int)
            value = f"{value_int} {blue_bars}{white_bars}"
            if gfm_supported:
                markdown_text += f"<tr><td>"
                markdown_text += f"{emoji}&nbsp;<strong>{key_nice}</strong>: {value}"
                markdown_text += f"</td></tr>\n"
            else:
                markdown_text += f"### {emoji} {key_nice}: {value}\n\n"
        elif 'relevant tests' in key_nice.lower():
            value = str(value).strip().lower()
            if gfm_supported:
                markdown_text += f"<tr><td>"
                if not is_value_no(value):
                    markdown_text += f"{emoji}&nbsp;<strong>PR contains tests</strong>"
                markdown_text += f"</td></tr>\n"
            else:
                if not is_value_no(value):
                    markdown_text += f"### {emoji} PR contains tests\n\n"
        elif 'security concerns' in key_nice.lower():
            if gfm_supported:
                markdown_text += f"<tr><td>"
                if is_value_no(value):
                    markdown_text += f"{emoji}&nbsp;<strong>No security concerns identified</strong>"
                else:
                    markdown_text += f"{emoji}&nbsp;<strong>Security concerns</strong><br><br>\n\n"
                    value = emphasize_header(value.strip())
                    markdown_text += f"{value}"
                markdown_text += f"</td></tr>\n"
            else:
                if is_value_no(value):
                    markdown_text += f'### {emoji} No security concerns identified\n\n'
                else:
                    markdown_text += f"### {emoji} Security concerns\n\n"
                    value = emphasize_header(value.strip(), only_markdown=True)
                    markdown_text += f"{value}\n\n"
        elif 'key issues to review' in key_nice.lower():
            # value is a list of issues
            if is_value_no(value):
                if gfm_supported:
                    markdown_text += f"<tr><td>"
                    markdown_text += f"{emoji}&nbsp;<strong>No major issues detected</strong>"
                    markdown_text += f"</td></tr>\n"
                else:
                    markdown_text += f"### {emoji} No major issues detected\n\n"
            else:
                issues = value
                if gfm_supported:
                    markdown_text += f"<tr><td>"
                    # markdown_text += f"{emoji}&nbsp;<strong>{key_nice}</strong><br><br>\n\n"
                    markdown_text += f"{emoji}&nbsp;<strong>Recommended focus areas for review</strong><br><br>\n\n"
                else:
                    markdown_text += f"### {emoji} Recommended focus areas for review\n\n#### \n"
                for i, issue in enumerate(issues):
                    try:
                        if not issue or not isinstance(issue, dict):
                            continue
                        relevant_file = issue.get('relevant_file', '').strip()
                        issue_header = issue.get('issue_header', '').strip()
                        if issue_header.lower() == 'possible bug':
                            issue_header = 'Possible Issue'  # Make the header less frightening
                        issue_content = issue.get('issue_content', '').strip()
                        start_line = int(str(issue.get('start_line', 0)).strip())
                        end_line = int(str(issue.get('end_line', 0)).strip())

                        relevant_lines_str = extract_relevant_lines_str(end_line, files, relevant_file, start_line, dedent=True)
                        reference_link = None

                        if gfm_supported:
                            if reference_link is not None and len(reference_link) > 0:
                                if relevant_lines_str:
                                    issue_str = f"<details><summary><a href='{reference_link}'><strong>{issue_header}</strong></a>\n\n{issue_content}\n</summary>\n\n{relevant_lines_str}\n\n</details>"
                                else:
                                    issue_str = f"<a href='{reference_link}'><strong>{issue_header}</strong></a><br>{issue_content}"
                            else:
                                issue_str = f"<strong>{issue_header}</strong><br>{issue_content}"
                        else:
                            if reference_link is not None and len(reference_link) > 0:
                                issue_str = f"[**{issue_header}**]({reference_link})\n\n{issue_content}\n\n"
                            else:
                                issue_str = f"**{issue_header}**\n\n{issue_content}\n\n"
                        markdown_text += f"{issue_str}\n\n"
                    except Exception as e:
                        get_logger().exception(f"Failed to process 'Recommended focus areas for review': {e}")
                if gfm_supported:
                    markdown_text += f"</td></tr>\n"
        else:
            if gfm_supported:
                markdown_text += f"<tr><td>"
                markdown_text += f"{emoji}&nbsp;<strong>{key_nice}</strong>: {value}"
                markdown_text += f"</td></tr>\n"
            else:
                markdown_text += f"### {emoji} {key_nice}: {value}\n\n"

    if gfm_supported:
        markdown_text += "</table>\n"

    return markdown_text

def set_file_languages(diff_files) -> List[FilePatchInfo]:
    try:
        # if the language is already set, do not change it
        if hasattr(diff_files[0], 'language') and diff_files[0].language:
            return diff_files

        # map file extensions to programming languages
        language_extension_map_org = LANGUAGE_EXTENSION_MAP
        extension_to_language = {}
        for language, extensions in language_extension_map_org.items():
            for ext in extensions:
                extension_to_language[ext] = language
        for file in diff_files:
            language_name = "txt"
            if file.filename is not None:
                extension_s = '.' + file.filename.rsplit('.')[-1]
                if extension_s and (extension_s in extension_to_language):
                    language_name = extension_to_language[extension_s]
            file.language = language_name.lower()
    except Exception as e:
        get_logger().exception(f"Failed to set file languages: {e}")

    return diff_files

def extract_hunk_lines_from_patch(patch: str, file_name, line_start, line_end, side, remove_trailing_chars: bool = True) -> tuple[str, str]:
    try:
        patch_with_lines_str = f"\n\n## File: '{file_name.strip()}'\n\n"
        selected_lines = ""
        patch_lines = patch.splitlines()
        RE_HUNK_HEADER = re.compile(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)")
        match = None
        start1, size1, start2, size2 = -1, -1, -1, -1
        skip_hunk = False
        selected_lines_num = 0
        for line in patch_lines:
            if 'no newline at end of file' in line.lower():
                continue

            if line.startswith('@@'):
                skip_hunk = False
                selected_lines_num = 0
                header_line = line

                match = RE_HUNK_HEADER.match(line)

                section_header, size1, size2, start1, start2 = extract_hunk_headers(match)

                # check if line range is in this hunk
                if side.lower() == 'left':
                    # check if line range is in this hunk
                    if not (start1 <= line_start <= start1 + size1):
                        skip_hunk = True
                        continue
                elif side.lower() == 'right':
                    if not (start2 <= line_start <= start2 + size2):
                        skip_hunk = True
                        continue
                patch_with_lines_str += f'\n{header_line}\n'

            elif not skip_hunk:
                if side.lower() == 'right' and line_start <= start2 + selected_lines_num <= line_end:
                    selected_lines += line + '\n'
                if side.lower() == 'left' and start1 <= selected_lines_num + start1 <= line_end:
                    selected_lines += line + '\n'
                patch_with_lines_str += line + '\n'
                if not line.startswith('-'): # currently we don't support /ask line for deleted lines
                    selected_lines_num += 1
    except Exception as e:
        get_logger().error(f"Failed to extract hunk lines from patch: {e}", artifact={"traceback": traceback.format_exc()})
        return "", ""

    if remove_trailing_chars:
        patch_with_lines_str = patch_with_lines_str.rstrip()
        selected_lines = selected_lines.rstrip()

    return patch_with_lines_str, selected_lines

def extract_relevant_lines_str(end_line, files, relevant_file, start_line, dedent=False) -> str:
    """
    Finds 'relevant_file' in 'files', and extracts the lines from 'start_line' to 'end_line' string from the file content.
    """
    try:
        relevant_lines_str = ""
        if files:
            files = set_file_languages(files)
            for file in files:
                if file.filename is not None and file.filename.strip() == relevant_file:
                    if not file.head_file:
                        # as a fallback, extract relevant lines directly from patch
                        patch = file.patch
                        get_logger().info(f"No content found in file: '{file.filename}' for 'extract_relevant_lines_str'. Using patch instead")
                        _, selected_lines = extract_hunk_lines_from_patch(patch, file.filename, start_line, end_line,side='right')
                        if not selected_lines:
                            get_logger().error(f"Failed to extract relevant lines from patch: {file.filename}")
                            return ""
                        # filter out '-' lines
                        relevant_lines_str = ""
                        for line in selected_lines.splitlines():
                            if line.startswith('-'):
                                continue
                            relevant_lines_str += line[1:] + '\n'
                    else:
                        relevant_file_lines = file.head_file.splitlines()
                        relevant_lines_str = "\n".join(relevant_file_lines[start_line - 1:end_line])

                    if dedent and relevant_lines_str:
                        # Remove the longest leading string of spaces and tabs common to all lines.
                        relevant_lines_str = textwrap.dedent(relevant_lines_str)
                    relevant_lines_str = f"```{file.language}\n{relevant_lines_str}\n```"
                    break

        return relevant_lines_str
    except Exception as e:
        get_logger().exception(f"Failed to extract relevant lines: {e}")
        return ""


def parse_review(prediction: str, files: list[FilePatchInfo]) -> str:
    """
    Prepare the PR review by processing the AI prediction and generating a markdown-formatted text that summarizes
    the feedback.
    """
    first_key = 'review'
    last_key = 'security_concerns'
    data = load_yaml(prediction.strip(),
                        keys_fix_yaml=["ticket_compliance_check", "estimated_effort_to_review_[1-5]:", "security_concerns:", "key_issues_to_review:",
                                    "relevant_file:", "relevant_line:", "suggestion:"],
                        first_key=first_key, last_key=last_key)

    # move data['review'] 'key_issues_to_review' key to the end of the dictionary
    if 'key_issues_to_review' in data['review']:
        key_issues_to_review = data['review'].pop('key_issues_to_review')
        data['review']['key_issues_to_review'] = key_issues_to_review

    markdown_text = convert_to_markdown_v2(data, False, files=files)

    if markdown_text == None or len(markdown_text) == 0:
        markdown_text = ""

    return markdown_text


def generate_task_base(args, item, file_lock) -> str | None:
    """Sends the diff and context to OpenAI for review using a structured prompt."""

    repo_path = f"/SWRBench/data/projects/{item['repo'].replace('/', '__')}/{item['instance_id']}"
    title = item['pr_title']
    description = item['pr_statement']
    base_branch_name = 'base_branch'
    target_branch_name = 'branch_under_review'
    
    token_handler = TokenHandler(args.model, SYSTEM_PROMPT, USER_PROMPT)
    diff_content, diff_files = get_pr_diff_old(repo_path, base_branch_name, target_branch_name, 
                               token_handler, args.model, add_line_numbers_to_hunks=True)
    # diff_content = get_pr_diff(item)
    # Consider if review is needed if only commit messages changed but no code diff
    if not diff_content:
        return "No code changes or significant commit messages detected, nothing to review."

    system_prompt = SYSTEM_PROMPT
    date = datetime.now().strftime("%Y-%m-%d")
    # user_prompt = USER_PROMPT.format(
    #     date=date,
    #     title=title,
    #     branch=target_branch_name,
    #     description=description,
    #     diff_content=diff_content or "[No code changes in diff]",
    # )
    user_prompt = USER_PROMPT.format(
        date=date,
        title=base_branch_name,
        branch=target_branch_name,
        diff_content=diff_content.strip() or "[No code changes in diff]",
    )

    get_logger().info(f"Sending request for instance {item['instance_id']} ...")
    get_logger().debug(f"Prompt: \n{user_prompt}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = run_chat(
        model=args.model, 
        messages=messages, 
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )
    if response is None:
        get_logger().info(f"Failed to get response for instance {item['instance_id']}")
        return
    
    get_logger().debug(f"Received response for instance {item['instance_id']}: {response}")
    messages.append({"role": "assistant", "content": response})
    review = parse_review(response, diff_files)
    
    with file_lock:
        with open(args.output_file, "a") as f:
            f.write(json.dumps({
                "instance_id": item["instance_id"],
                "review": review,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": response,
            }) + "\n")


def generate_task_refine_v1(args, item, file_lock) -> str | None:
    """Sends the diff and context to OpenAI for review using a structured prompt."""

    repo_path = f"/SWRBench/data/projects/{item['repo'].replace('/', '__')}/{item['instance_id']}"
    title = item['pr_title']
    description = item['pr_statement']
    base_branch_name = 'base_branch'
    target_branch_name = 'branch_under_review'
    
    token_handler = TokenHandler(args.model, SYSTEM_PROMPT, USER_PROMPT)
    diff_content, diff_files = get_pr_diff_old(repo_path, base_branch_name, target_branch_name, 
                               token_handler, args.model, add_line_numbers_to_hunks=True)
    # diff_content = get_pr_diff(item)
    # Consider if review is needed if only commit messages changed but no code diff
    if not diff_content:
        return "No code changes or significant commit messages detected, nothing to review."

    system_prompt = SYSTEM_PROMPT
    date = datetime.now().strftime("%Y-%m-%d")
    # user_prompt = USER_PROMPT.format(
    #     date=date,
    #     title=title,
    #     branch=target_branch_name,
    #     description=description,
    #     diff_content=diff_content or "[No code changes in diff]",
    # )
    user_prompt = USER_PROMPT.format(
        date=date,
        title=base_branch_name,
        branch=target_branch_name,
        diff_content=diff_content.strip() or "[No code changes in diff]",
    )

    get_logger().info(f"Sending request for instance {item['instance_id']} ...")
    get_logger().debug(f"Prompt: \n{user_prompt}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    responses = []
    for _ in range(3):
        response = run_chat(
            model=args.model, 
            messages=messages, 
            temperature=args.temperature, 
            max_tokens=args.max_tokens
        )
        if response is None:
            get_logger().info(f"Failed to get response for instance {item['instance_id']}")
            return
        responses.append(response)
    
    refine_prompt = (
        f"**Goal:** Generate a high-quality, consolidated code review report by analyzing the provided Pull Request (PR) details and critically evaluating three draft review reports.\n\n"
        f"**Role:** Act as an expert code reviewer tasked with synthesizing the most accurate, actionable, and constructive feedback possible.\n\n"
        f"**Process:**\n"
        f"1.  **Deeply Understand the Pull Request:** Carefully read and comprehend the `<Pull Request Details>` below. Identify the PR's objectives, the specific changes made (even if inferred from the description), and the context.\n"
        f"2.  **Critically Evaluate Draft Reports:** Analyze each of the three `<Draft Review Report>` sections provided. For *every* comment or point raised in these drafts, assess it based on the following criteria:\n"
        f"    *   **Accuracy:** Does the comment accurately reflect the code or changes described/implied in the `<Pull Request Details>`? **Discard any points that are factually incorrect or misinterpret the PR.**\n"
        f"    *   **Relevance & Significance:** Is the comment relevant to the changes in the PR? Does it address potential bugs, design flaws, security concerns, performance issues, deviations from standards, or areas for meaningful improvement? Prioritize significant points over minor nitpicks unless specifically requested.\n"
        f"    *   **Clarity & Actionability:** Is the comment clear, specific, and easy to understand? Does it suggest a concrete action the author can take? Vague or ambiguous comments should be refined for clarity or discarded if they cannot be made actionable.\n"
        f"    *   **Constructiveness:** Is the tone helpful and professional? Avoid points that are overly negative or unhelpful.\n"
        f"    *   **Redundancy:** Identify points that are essentially duplicates across the different drafts.\n"
        f"3.  **Synthesize the Final Consolidated Report:** Construct a single, coherent, and improved review report by performing the following:\n"
        f"    *   **Select Validated Points:** Only include points from the drafts that you have verified as accurate, relevant, and actionable based *strictly* on the `<Pull Request Details>`.\n"
        f"    *   **Consolidate & Refine:** Merge duplicate or very similar points into a single, well-phrased comment. Rephrase comments where necessary to improve clarity, conciseness, and maintain a constructive tone.\n"
        f"    *   **Structure Logically:** Organize the feedback in a clear and logical manner. Consider grouping comments by severity (e.g., Major Concerns, Suggestions, Minor Nitpicks), by file/module, or by theme. A brief introductory summary of the review might be beneficial.\n"
        f"    *   **Ensure Completeness (Based on Drafts):** Aim to cover the valid, important points raised across all three drafts, without introducing new points not derived from the drafts or the PR details.\n"
        f"    *   **Maintain Professional Tone:** The final report must be professional, objective, and focused on improving the code quality.\n\n"
        f"**Input Data:**\n\n"
        f"<Start of Pull Request Details>\n"
        f"{user_prompt}\n"
        f"<End of Pull Request Details>\n\n"
        f"<Start of Draft Review Report 1>\n"
        f"{responses[0]}\n"
        f"<End of Draft Review Report 1>\n\n"
        f"<Start of Draft Review Report 2>\n"
        f"{responses[1]}\n"
        f"<End of Draft Review Report 2>\n\n"
        f"<Start of Draft Review Report 3>\n"
        f"{responses[2]}\n"
        f"<End of Draft Review Report 3>\n\n"
        f"**Task:** Now, generate the final, consolidated, and improved code review report based *only* on the provided PR details and the critically evaluated points from the draft reports, following all instructions above.\n"
        "Response (should be a valid YAML, and nothing else):\n"
        "```yaml\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": refine_prompt},
    ]
    response = run_chat(
        model=args.model, 
        messages=messages, 
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )
    
    get_logger().debug(f"Received response for instance {item['instance_id']}: {response}")
    messages.append({"role": "assistant", "content": response})
    review = response
    review = parse_review(response, diff_files)
    
    with file_lock:
        with open(args.output_file, "a") as f:
            f.write(json.dumps({
                "instance_id": item["instance_id"],
                "review": review,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "refine_prompt": refine_prompt,
                "response": response,
                "responses": responses,
            }) + "\n")


def generate_task_refine(args, item, file_lock) -> str | None:
    """Sends the diff and context to OpenAI for review using a structured prompt."""
    responses_dict = {}
    
    assert args.review_paths is not None, "review_paths is required"
    review_paths = args.review_paths.split(",")
    assert len(review_paths) > 0, "review_paths must contain at least one path"
    for review_path in review_paths:
        with open(review_path, "r") as f:
            reviews = load_jsonl(review_path)
            for review_item in reviews:
                if review_item["instance_id"] not in responses_dict:
                    responses_dict[review_item["instance_id"]] = []
                review = review_item["review"]
                response = review_item["response"]
                if "<think>" in response and "</think>" in response:
                    response = response.split("</think>")[1]
                responses_dict[review_item["instance_id"]].append({
                    "review": review,
                    "response": response,
                })
    repo_path = f"/SWRBench/data/projects/{item['repo'].replace('/', '__')}/{item['instance_id']}"
    title = item['pr_title']
    base_branch_name = 'base_branch'
    target_branch_name = 'branch_under_review'
    
    if not os.path.exists(repo_path):
        get_logger().warning(f"Repository path does not exist: {repo_path}")
        return "Repository path does not exist."
    
    system_prompt = SYSTEM_PROMPT
    try_times = 0
    while try_times < 5:
        token_handler = TokenHandler(args.model, SYSTEM_PROMPT, USER_PROMPT)
        diff_content, diff_files = get_pr_diff_old(repo_path, base_branch_name, target_branch_name, 
                                   token_handler, args.model, add_line_numbers_to_hunks=True)
        # diff_content = get_pr_diff(item)
        # Consider if review is needed if only commit messages changed but no code diff
        if not diff_content:
            get_logger().warning(f"No code changes or significant commit messages detected, nothing to review for instance {item['instance_id']}")
            return "No code changes or significant commit messages detected, nothing to review."

        responses = responses_dict[item["instance_id"]]
        random.shuffle(responses)
        responses_str = "\n\n".join([f"<Start of Draft Review Report {i+1}>\n{res['response']}\n<End of Draft Review Report {i+1}>\n" for i, res in enumerate(responses)])
        refine_prompt = REFINE_PROMPT.format(
            title=title,
            branch=target_branch_name,
            review_report=responses_str,
            diff_content=diff_content,
        )
        
        get_logger().info(f"Sending request for instance {item['instance_id']} ...")
        get_logger().debug(f"Prompt: \n{refine_prompt}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refine_prompt},
        ]
        response = run_chat(
            model=args.model, 
            messages=messages, 
            temperature=args.temperature, 
            max_retries=3,
        )
        
        if response is None:
            set_max_tokens(get_max_tokens(args.model) - 4000)
            get_logger().warning(f"Failed to get response for instance {item['instance_id']}, set max tokens to {get_max_tokens(args.model)}, try again...")
            try_times += 1
            continue
        break
    
    get_logger().debug(f"Received response for instance {item['instance_id']}: {response}")
    messages.append({"role": "assistant", "content": response})
    try:
        if "<think>" in response and "</think>" in response:
            response = response.split("</think>")[1]
        review = parse_review(response, diff_files)
    except Exception as e:
        get_logger().exception(f"Failed to parse review for instance {item['instance_id']}: {e}")
        review = "ERROR"
    # import pdb; pdb.set_trace()
    with file_lock:
        with open(args.output_file, "a") as f:
            f.write(json.dumps({
                "instance_id": item["instance_id"],
                "review": review,
                "system_prompt": system_prompt,
                "refine_prompt": refine_prompt,
                "response": response,
            }) + "\n")

# --- Main Execution ---

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path: str, data: List[Dict]):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_dataset(args):
    dataset = load_jsonl(args.dataset_file)
    if args.instance_ids:
        dataset = [i for i in dataset if i["instance_id"] in args.instance_ids]
    if args.ignore_ids:
        dataset = [i for i in dataset if i["instance_id"] not in args.ignore_ids]
    
    task_dataset = []
    for item in dataset:
        task_dataset.append(item)

    return task_dataset

def generate(args):
    dataset = load_dataset(args)
    
    print("="*100)
    print(f"args: {args}")
    print("="*100)
    print(f"Evaluating {len(dataset)} instances")
    print(f"Dataset File: {args.dataset_file}")
    print(f"Model: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Output File: {args.output_file}")
    print("="*100)
    
    print(f"Copying dataset {args.num_samples} times.")
    dataset = dataset * args.num_samples
    random.seed(42)
    random.shuffle(dataset)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.path.exists(args.output_file) and not args.clean:
        previous_results = load_jsonl(args.output_file)
        previous_results = {i["instance_id"]: i for i in previous_results if i["review"] != "ERROR"}
        dataset = [i for i in dataset if i["instance_id"] not in previous_results]
        save_jsonl(args.output_file, list(previous_results.values()))
    else:
        with open(args.output_file, "w") as f:
            f.write("")
    
    log_file = args.output_file + ".log"
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    
    global LOGGER, MAX_TOKENS
    
    LOGGER = logging.getLogger(__name__)
    MAX_TOKENS = args.max_tokens if args.max_tokens is not None else None
    file_lock = threading.Lock()
    
    def process_item(item):
        # import pdb; pdb.set_trace()
        if args.refine:
            generate_task_refine(args, item, file_lock)
        else:
            generate_task_base(args, item, file_lock)
        return True

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        
        for _ in tqdm(as_completed(futures), total=len(dataset), desc="Processing"):
            pass
    
    # for item in tqdm(dataset, total=len(dataset), desc="Processing"):
    #     process_item(item)
    
    print(f"Finished processing {len(dataset)} instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=str, help="Path to dataset file")
    parser.add_argument("--review-paths", type=str, default=None, help="Review paths")
    parser.add_argument("--output-file", type=str, help="Path to output file")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--max-tokens", default=32000, type=int, help="Max tokens")
    parser.add_argument("--temperature", default=0.2, type=float, help="Temperature")
    parser.add_argument("--num-samples", default=1, type=int, help="Number of samples")
    parser.add_argument("--instance-ids", nargs="+", help="Instance ids")
    parser.add_argument("--ignore-ids", nargs="+", help="Ignore instance ids")
    parser.add_argument("--num-threads", default=1, type=int, help="Number of threads")
    parser.add_argument("--max-prompt-length", default=20000, type=int, help="Max prompt length")
    parser.add_argument("--clean", action="store_true", help="Clean output file")
    parser.add_argument("--refine", action="store_true", help="Refine review")
    args = parser.parse_args()
    
    generate(args)