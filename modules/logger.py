



class md_logger:
    def __init__(self, log_file=None):

        if log_file is None:            
            self.log = open('log.md', 'w')
        else:
            self.log = open(log_file, 'w')
        self.print_out_ = True
    
    def print_out(self, to_print):
        if self.print_out_ :
            print(to_print)

    def img(self, img_name):
        self.log.write('!['+ img_name + "](" + img_name + ')\n')

    def table(self, table):
        self.log.write(table.to_markdown() + '\n')
        self.log.write('\n')


    def code(self, code):
        self.log.write('```python\n' + code + '\n```\n')

    def title(self, title):
        self.log.write('# ' + title + '\n')
        self.print_out(title)


    def subtitle(self, subtitle):
        self.log.write('## ' + subtitle + '\n')
        self.print_out(subtitle)


    def text(self, text):
        self.log.write(text + '\n')
        self.print_out(text)
        

    def bold(self, text):
        self.log.write('**' + text + '**\n')

    def italic(self, text):
        self.log.write('*' + text + '*\n')        

    def file(self, file):
        with open(file, 'r') as f:
            self.log.write(f.read())
            self.log.write('\n')

    def link(self, link, text=""):
        if text != "":
            self.log.write('[' + text + '](' + link + ')\n')
        else:
            self.log.write('[' + link + '](' + link + ')\n')

    def footnote_link(self, link, text=""):
        if text != "":
            self.log.write('[^' + text + ']')
        else:
            self.log.write('[^' + link + ']')
            
    def footnote(self, text, link):
        self.log.write('[^' + link + ']: ' + text + '\n')

    def close(self):
        self.log.close()