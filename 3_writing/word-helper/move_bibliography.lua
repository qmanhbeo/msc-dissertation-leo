-- move_bibliography.lua -- Word-side bibliography relocation for the docx build.
--
-- Why: pandoc's --citeproc always appends the bibliography (Div class
-- "references") at the very end of the document, ignoring the source position
-- of \printbibliography. In dissertation.tex the bibliography comes BEFORE
-- \appendix (Conclusion -> Reference list -> Appendix), which is what the
-- compiled PDF shows. This filter restores that order in the docx by moving
-- the citeproc bibliography Div to just before the first appendix section and
-- injecting an unnumbered "Reference list" heading there (matching the PDF's
-- unnumbered \printbibliography chapter).
--
-- Ordering: must run AFTER word_section_numbers.lua (so the appendix header is
-- already labelled "A") and AFTER citeproc (so the bibliography Div exists),
-- but BEFORE lof_lot.lua (which wraps sections and would defeat the
-- top-level appendix-header detection below).

function Pandoc(doc)
  -- Locate the citeproc-generated bibliography Div.
  local bib_idx = nil
  for i, b in ipairs(doc.blocks) do
    if b.t == "Div" then
      for _, c in ipairs(b.attr.classes) do
        if c == "references" then bib_idx = i end
      end
    end
  end
  if not bib_idx then return doc end

  -- Locate the first appendix section: a level-1 header whose label is a
  -- letter (A..), not a digit (1..6 main chapters). word_section_numbers.lua
  -- has already prefixed these labels.
  local app_idx = nil
  for i, b in ipairs(doc.blocks) do
    if b.t == "Header" and b.level == 1 then
      local f = b.content[1]
      if f and f.t == "Str" and f.text:match("^%u") and not f.text:match("^%d") then
        app_idx = i
        break
      end
    end
  end
  if not app_idx then return doc end

  -- Unnumbered "Reference list" heading, mirroring the PDF's
  -- \printbibliography[title={Reference list}] (a chapter* heading).
  local heading = pandoc.Header(1, pandoc.Inlines { pandoc.Str("Reference list") })
  heading.attr.classes:insert("unnumbered")

  -- Force the Reference list onto its own page (pandoc ignores LaTeX
  -- \clearpage, so inject the OpenXML page break here to match the PDF).
  local PAGE_BREAK = pandoc.RawBlock("openxml",
    '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

  local bib = doc.blocks[bib_idx]
  table.remove(doc.blocks, bib_idx)
  table.insert(doc.blocks, app_idx, PAGE_BREAK)
  table.insert(doc.blocks, app_idx + 1, heading)
  table.insert(doc.blocks, app_idx + 2, bib)

  -- Apply hanging indent (0.5 in = 720 twips) to every bibliography paragraph
  -- so each entry's first line is flush-left and continuation lines indent.
  if bib.t == "Div" then
    for i, block in ipairs(bib.content) do
      if block.t == "Para" then
        block.attr = block.attr or pandoc.Attr()
        block.attr.attributes["firstLine"] = "-720"
      end
    end
  end

  return doc
end
