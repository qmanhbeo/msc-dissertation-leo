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

  local bib = doc.blocks[bib_idx]
  table.remove(doc.blocks, bib_idx)
  table.insert(doc.blocks, app_idx, heading)
  table.insert(doc.blocks, app_idx + 1, bib)
  return doc
end
