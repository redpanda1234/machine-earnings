(TeX-add-style-hook
 "proposal"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "margin=1in" "tmargin=1.25in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "geometry"
    "fancyhdr"
    "lastpage"))
 :latex)

