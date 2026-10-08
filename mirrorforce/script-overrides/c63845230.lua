-- Eater of Millions
-- MirrorForce runtime-equivalent performance override.
--
-- The upstream procedure calls CheckSubGroup over every eligible card in the
-- hand, field and Extra Deck with max=#g.  Eater commonly sees 20--40 cards,
-- so the generic 2^N completion search can spin for hours.  MZONE availability
-- depends only on cards that will leave the field.  Search that bounded field
-- subset exactly; cards outside the field are interchangeable count fillers.
function c63845230.initial_effect(c)
	c:EnableReviveLimit()
	-- special summon condition
	local e1=Effect.CreateEffect(c)
	e1:SetType(EFFECT_TYPE_SINGLE)
	e1:SetProperty(EFFECT_FLAG_CANNOT_DISABLE+EFFECT_FLAG_UNCOPYABLE)
	e1:SetCode(EFFECT_SPSUMMON_CONDITION)
	c:RegisterEffect(e1)
	-- special summon
	local e2=Effect.CreateEffect(c)
	e2:SetType(EFFECT_TYPE_FIELD)
	e2:SetProperty(EFFECT_FLAG_CANNOT_DISABLE+EFFECT_FLAG_UNCOPYABLE)
	e2:SetCode(EFFECT_SPSUMMON_PROC)
	e2:SetRange(LOCATION_HAND)
	e2:SetCondition(c63845230.spcon)
	e2:SetTarget(c63845230.sptg)
	e2:SetOperation(c63845230.spop)
	c:RegisterEffect(e2)
	-- atk/def
	local e3=Effect.CreateEffect(c)
	e3:SetType(EFFECT_TYPE_SINGLE)
	e3:SetProperty(EFFECT_FLAG_SINGLE_RANGE)
	e3:SetCode(EFFECT_UPDATE_ATTACK)
	e3:SetRange(LOCATION_MZONE)
	e3:SetValue(c63845230.val)
	c:RegisterEffect(e3)
	local e4=e3:Clone()
	e4:SetCode(EFFECT_UPDATE_DEFENSE)
	c:RegisterEffect(e4)
	-- cannot release or use as material
	local e5=Effect.CreateEffect(c)
	e5:SetType(EFFECT_TYPE_SINGLE)
	e5:SetProperty(EFFECT_FLAG_SINGLE_RANGE)
	e5:SetCode(EFFECT_UNRELEASABLE_SUM)
	e5:SetRange(LOCATION_MZONE)
	e5:SetValue(1)
	c:RegisterEffect(e5)
	local e6=e5:Clone()
	e6:SetCode(EFFECT_UNRELEASABLE_NONSUM)
	c:RegisterEffect(e6)
	local e7=e5:Clone()
	e7:SetCode(EFFECT_CANNOT_BE_FUSION_MATERIAL)
	e7:SetValue(c63845230.fuslimit)
	c:RegisterEffect(e7)
	local e8=e5:Clone()
	e8:SetCode(EFFECT_CANNOT_BE_SYNCHRO_MATERIAL)
	c:RegisterEffect(e8)
	local e9=e5:Clone()
	e9:SetCode(EFFECT_CANNOT_BE_XYZ_MATERIAL)
	c:RegisterEffect(e9)
	-- banish the battle opponent face-down
	local ea=Effect.CreateEffect(c)
	ea:SetDescription(aux.Stringid(63845230,0))
	ea:SetCategory(CATEGORY_REMOVE)
	ea:SetType(EFFECT_TYPE_SINGLE+EFFECT_TYPE_TRIGGER_O)
	ea:SetCode(EVENT_BATTLE_START)
	ea:SetCountLimit(1)
	ea:SetTarget(c63845230.rmtg)
	ea:SetOperation(c63845230.rmop)
	c:RegisterEffect(ea)
end

function c63845230.fuslimit(e,c,sumtype)
	return sumtype==SUMMON_TYPE_FUSION
end

function c63845230.costfilter(c)
	return c:IsAbleToRemoveAsCost()
end

function c63845230.zoneok(g,tp)
	return #g>=5 and Duel.GetMZoneCount(tp,g)>0
end

-- Exact bounded DFS over optional on-field additions.  ``base`` is already
-- selected; ``outside`` counts remaining hand/Extra cards, which can raise the
-- final group to five without changing MZONE availability.
function c63845230.fieldcompletion(fields,chosen,base,outside,tp)
	if #base+#chosen+outside>=5 then
		local tg=base:Clone()
		tg:Merge(chosen)
		if Duel.GetMZoneCount(tp,tg)>0 then return true end
	end
	if #fields==0 or #base+#chosen+#fields+outside<5 then return false end
	local tc=fields:GetFirst()
	local rest=fields:Clone()
	rest:RemoveCard(tc)
	if c63845230.fieldcompletion(rest,chosen,base,outside,tp) then
		return true
	end
	chosen:AddCard(tc)
	local ok=c63845230.fieldcompletion(rest,chosen,base,outside,tp)
	chosen:RemoveCard(tc)
	return ok
end

function c63845230.cancomplete(base,all,tp)
	local remaining=all-base
	if #base+#remaining<5 then return false end
	local fields=remaining:Filter(Card.IsLocation,nil,LOCATION_ONFIELD)
	local outside=#remaining-#fields
	return c63845230.fieldcompletion(
		fields,Group.CreateGroup(),base,outside,tp)
end

function c63845230.pickfilter(c,sg,all,tp)
	local tg=sg:Clone()
	tg:AddCard(c)
	return c63845230.cancomplete(tg,all,tp)
end

function c63845230.spcon(e,c)
	if c==nil then return true end
	local tp=c:GetControler()
	local g=Duel.GetMatchingGroup(c63845230.costfilter,tp,
		LOCATION_HAND+LOCATION_ONFIELD+LOCATION_EXTRA,0,c)
	return c63845230.cancomplete(Group.CreateGroup(),g,tp)
end

function c63845230.sptg(e,tp,eg,ep,ev,re,r,rp,chk,c)
	local g=Duel.GetMatchingGroup(c63845230.costfilter,tp,
		LOCATION_HAND+LOCATION_ONFIELD+LOCATION_EXTRA,0,c)
	local sg=Group.CreateGroup()
	local forced=Duel.GrabSelectedCard()
	if forced then sg:Merge(forced) end
	while #sg<#g do
		local finish=c63845230.zoneok(sg,tp)
		local cg=(g-sg):Filter(c63845230.pickfilter,nil,sg,g,tp)
		if #cg==0 then break end
		Duel.Hint(HINT_SELECTMSG,tp,HINTMSG_REMOVE)
		local tc=cg:SelectUnselect(sg,tp,finish,not finish,1,1)
		if not tc then break end
		if sg:IsContains(tc) then
			if not forced or not forced:IsContains(tc) then sg:RemoveCard(tc) end
		else
			sg:AddCard(tc)
		end
	end
	if not c63845230.zoneok(sg,tp) then return false end
	sg:KeepAlive()
	e:SetLabelObject(sg)
	return true
end

function c63845230.spop(e,tp,eg,ep,ev,re,r,rp,c)
	local g=e:GetLabelObject()
	Duel.Remove(g,POS_FACEDOWN,REASON_SPSUMMON)
	g:DeleteGroup()
end

function c63845230.val(e,c)
	return Duel.GetMatchingGroupCount(Card.IsFacedown,0,
		LOCATION_REMOVED,LOCATION_REMOVED,nil)*100
end

function c63845230.rmtg(e,tp,eg,ep,ev,re,r,rp,chk)
	local tc=e:GetHandler():GetBattleTarget()
	if chk==0 then
		return tc and tc:IsControler(1-tp)
			and tc:IsAbleToRemove(tp,POS_FACEDOWN)
	end
	Duel.SetOperationInfo(0,CATEGORY_REMOVE,tc,1,0,0)
end

function c63845230.rmop(e,tp,eg,ep,ev,re,r,rp)
	local tc=e:GetHandler():GetBattleTarget()
	if tc:IsRelateToBattle() then
		Duel.Remove(tc,POS_FACEDOWN,REASON_EFFECT)
	end
end
