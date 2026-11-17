# Commit do código, preenchido pelo próprio git ao gerar um pacote com
#   git archive --format=zip -o codigo.zip HEAD
# (atributo export-subst em .gitattributes). Num clone o marcador fica como
# está e utils._git_revision usa 'git rev-parse' em vez deste arquivo.
COMMIT = "$Format:%H$"
