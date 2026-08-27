-- lof_lot.lua -- List of Figures / List of Tables + caption numbering for the
-- docx build; Word-side counterpart of \listoffigures/\listoftables.
--
-- * Counts TOP-LEVEL Figure/Table blocks only. Nested subfigures (the H1a-d
--   panels inside the H1-grid figure) are excluded, so numbering matches the
--   LaTeX counters (verified: all labeled-float ref values equal the PDF's
--   own .aux values).
-- * Prefixes body captions "Figure N: " / "Table N: " (LaTeX default form).
-- * Injects [ToC, List of Figures, List of Tables] immediately before the
--   first Header, i.e. after the abstract block, mirroring the PDF front
--   matter order (abstract -> ToC -> LoF -> LoT). pandoc's own --toc always
--   emits the field at the very top of the docx and cannot be reordered, so
--   --toc is dropped from build_word.sh and the ToC is re-injected here as
--   the verbatim SDT pandoc itself generates.
--
-- Run AFTER word_section_numbers.lua: the two filters touch disjoint node
-- kinds (headers/section-refs vs figures/tables).

local TOC_SDT = [[<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents" /><w:docPartUnique /></w:docPartObj></w:sdtPr><w:sdtContent><w:p><w:pPr><w:pStyle w:val="TOCHeading" /></w:pPr><w:r><w:t xml:space="preserve">Table of Contents</w:t></w:r></w:p><w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true" /><w:instrText xml:space="preserve">TOC \o &quot;1-3&quot; \h \z \u</w:instrText><w:fldChar w:fldCharType="separate" /><w:fldChar w:fldCharType="end" /></w:r></w:p></w:sdtContent></w:sdt>]]

local function caption_inlines(caption)
  if not caption then return pandoc.List() end
  -- pandoc 3.x: Figure/Table caption is a Caption object (.long = blocks);
  -- tolerate a plain block list too.
  local blocks = caption.long or caption
  local out = pandoc.List()
  for _, b in ipairs(blocks) do
    if b.content then
      for _, x in ipairs(b.content) do out:insert(x) end
    end
  end
  return out
end

-- Labeled tables arrive as Div(id=tab:..) wrapping the Table node (the Table
-- itself carries no id), so anchors and numbering must be taken from the Div.
local function table_carrier(block)
  if block.t == "Table" then
    return block, block.identifier
  end
  if block.t == "Div" then
    for _, inner in ipairs(block.content) do
      if inner.t == "Table" then
        return inner, block.identifier
      end
    end
  end
  return nil
end

-- custom-style must match the template style's NAME ("TOC Heading"), not its
-- styleId. pandoc resolves custom-style by w:name; requesting "TOCHeading"
-- (the id) misses and makes pandoc emit a second, BodyText-based, non-bold
-- TOCHeading style that shadows the template's bold Heading1-based one.
local function list_title(text)
  return pandoc.Div({pandoc.Para({pandoc.Str(text)})},
                    pandoc.Attr("", {}, {{"custom-style", "TOC Heading"}}))
end

local function entry_para(prefix, inlines, anchor)
  local content = {pandoc.Str(prefix .. " ")}
  for _, x in ipairs(inlines) do content[#content + 1] = x end
  local para = pandoc.Plain(content)
  if anchor ~= "" then
    local link = pandoc.Link(content, "#" .. anchor)
    return pandoc.Para({link})
  end
  return pandoc.Para(content)
end

function Pandoc(doc)
  local figs, tabs = {}, {}   -- {n, inlines, anchor} in document order

  -- Pass 1: number top-level floats and prefix their captions.
  for _, b in ipairs(doc.blocks) do
    if b.t == "Figure" then
      local n = #figs + 1
      local cap = caption_inlines(b.caption)
      table.insert(figs, {n = n, inlines = cap, anchor = b.identifier})
      local pre = {pandoc.Str("Figure " .. n .. ": ")}
      for _, x in ipairs(cap) do pre[#pre + 1] = x end
      b.caption = {pandoc.Plain(pre)}
    else
      local tbl, anchor = table_carrier(b)
      if tbl then
        local n = #tabs + 1
        local cap = caption_inlines(tbl.caption)
        table.insert(tabs, {n = n, inlines = cap, anchor = anchor})
        local pre = {pandoc.Str("Table " .. n .. ": ")}
        for _, x in ipairs(cap) do pre[#pre + 1] = x end
        tbl.caption = {pandoc.Plain(pre)}
      end
    end
  end

  if #figs == 0 and #tabs == 0 then
    error("lof_lot: no captioned figures/tables found; refusing to emit empty lists.")
  end

  -- Pass 2: assemble [ToC, LoF, LoT] and insert before the first Header.
  local front = pandoc.List()
  front:insert(pandoc.RawBlock("openxml", TOC_SDT))
  front:insert(list_title("List of Figures"))
  for _, f in ipairs(figs) do
    front:insert(entry_para("Figure " .. f.n .. ":", f.inlines, f.anchor))
  end
  front:insert(list_title("List of Tables"))
  for _, t in ipairs(tabs) do
    front:insert(entry_para("Table " .. t.n .. ":", t.inlines, t.anchor))
  end

  local out = pandoc.List()
  local placed = false
  for _, b in ipairs(doc.blocks) do
    if not placed and b.t == "Header" then
      for _, x in ipairs(front) do out:insert(x) end
      placed = true
    end
    out:insert(b)
  end
  if not placed then
    error("lof_lot: no Header found; no safe insertion point for front matter.")
  end
  doc.blocks = out
  return doc
end
