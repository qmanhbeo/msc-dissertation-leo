-- word_section_numbers.lua -- Word-side section numbering for the docx build.
--
-- Why: pandoc's LaTeX reader resolves \ref{} into literal numbers baked as
-- Link content, while the docx writer numbers headings only via -N (absent
-- here; its 3.1.3 prefixes come out unspaced anyway). So headings previously
-- rendered bare next to numbered refs. This filter owns ALL heading labels
-- so headings and in-text refs can never disagree:
--   * Heading 1-3 get hierarchical labels ("1", "2.3", "3.8.1"); after
--     N_MAIN_DECIMAL top-level sections they become letters A, B, ...
--     mirroring article-class \appendix numbering in the compiled PDF.
--   * Heading 4 (\paragraph) stays unnumbered (article default secnumdepth=3).
--   * Every Link marked reference-type=ref targeting a known header gets its
--     content recomputed from the same map; fig/tab/etc. targets untouched.
--
-- NOTE: written in global-handler form; pandoc 3.1.3's Lua engine does not
-- invoke `return { Pandoc = ... }` table-style filters.

local N_MAIN_DECIMAL = 6   -- Introduction..Conclusion are decimal in source;
                           -- everything after these is an appendix section.

local function letter_for(n)
  if n < 1 or n > 26 then
    error("word_section_numbers: top-level section index "
      .. tostring(n) .. " exceeds letter scheme; extend deliberately.")
  end
  return string.char(string.byte("A") + n - 1)
end

-- pandoc 3.1.3's walk_block dispatches handlers only to DESCENDANTS, never to
-- the element itself, and discards in-place mutations unless the mutated node
-- is RETURNED; wrapping in a throwaway Div makes top-level blocks visible and
-- the caller must reuse the walked element.
local function walk_wrapped(block, handlers)
  return pandoc.walk_block(pandoc.Div{block}, handlers).content[1]
end

function Pandoc(doc)
  local number_by_id = {}
  local c1, c2, c3 = 0, 0, 0
  local p1, p2 = "", ""   -- formatted parent labels at levels 1 and 2

  -- Pass 1: collect labels for every Header (recurses through Divs etc.).
  for _, b in ipairs(doc.blocks) do
    walk_wrapped(b, {
      Header = function(h)
        if h.level == 1 then
          c1 = c1 + 1
          c2, c3 = 0, 0
          p1 = c1 <= N_MAIN_DECIMAL and tostring(c1) or letter_for(c1 - N_MAIN_DECIMAL)
          number_by_id[h.identifier] = p1
        elseif h.level == 2 then
          c2 = c2 + 1
          c3 = 0
          p2 = p1 .. "." .. c2
          number_by_id[h.identifier] = p2
        elseif h.level == 3 then
          c3 = c3 + 1
          number_by_id[h.identifier] = p2 .. "." .. c3
        else
          -- Level >= 4 (\paragraph) is unnumbered in the PDF (secnumdepth=3),
          -- but the LaTeX reader still assigns deep labels to it; refs to
          -- such labels must resolve to the nearest numbered ancestor so the
          -- Word output matches the PDF.
          if c1 > 0 then
            number_by_id[h.identifier] = (c2 > 0) and p2 or p1
          end
        end
        return nil
      end,
    })
  end

  -- Pass 2: inject label prefixes into headings and recompute ref contents.
  for i, b in ipairs(doc.blocks) do
    doc.blocks[i] = walk_wrapped(b, {
      Header = function(h)
        if h.level >= 1 and h.level <= 3 then
          local lbl = number_by_id[h.identifier]
          if lbl then
            local content = pandoc.List()
            content:insert(pandoc.Str(lbl))
            content:insert(pandoc.Space())
            for _, x in ipairs(h.content) do content:insert(x) end
            h.content = content
            return h
          end
        end
        return nil
      end,
      Link = function(l)
        if l.attributes["reference-type"] == "ref" then
          local lbl = number_by_id[l.attributes["reference"]]
          if lbl then
            l.content = {pandoc.Str(lbl)}
            return l
          end
        end
        return nil
      end,
    })
  end
  return doc
end
