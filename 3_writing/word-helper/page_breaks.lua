-- page_breaks.lua -- Insert page breaks before specific headers in the docx build.
--
-- pandoc's LaTeX reader ignores \clearpage/\newpage for docx, so this filter
-- injects a raw OpenXML page break before target headers.

local PAGE_BREAK = pandoc.RawBlock("openxml",
  '<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

-- Every chapter (main 1-6 and appendices A-J) must start on a fresh page per
-- the UB thesis guide. pandoc drops LaTeX \clearpage, so inject the page break
-- here, keyed on the numbered/lettered prefix word_section_numbers.lua prepends.
local targets = {
  ["1"] = true, ["2"] = true, ["3"] = true, ["4"] = true, ["5"] = true, ["6"] = true,
  ["A"] = true, ["B"] = true, ["C"] = true, ["D"] = true, ["E"] = true,
  ["F"] = true, ["G"] = true, ["H"] = true, ["I"] = true, ["J"] = true,
}

function Pandoc(doc)
  local out = pandoc.List()
  for _, b in ipairs(doc.blocks) do
    if b.t == "Header" and b.level == 1 then
      -- Check first content element for target text
      local first = b.content[1]
      if first and first.t == "Str" then
        local text = first.text
        -- Match "1 Introduction" or "A Appendix-title" etc.
        local letter_or_num = text:match("^([%dA-Za-z]+)")
        if letter_or_num and targets[letter_or_num] then
          out:insert(PAGE_BREAK)
        end
      end
    end
    out:insert(b)
  end
  doc.blocks = out
  return doc
end
